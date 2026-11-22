"""Export the Demucs htdemucs_6s model to ONNX for in-browser inference.

The STFT / inverse STFT are complex-valued and do not export cleanly, so they
are left out of the graph: the browser computes them in JavaScript (see
web/dsp.js). The exported graph takes the raw stereo segment plus its
spectrogram (real/imag stacked as channels) and returns the spectrogram-branch
and time-branch outputs separately.

Usage: python tools/export_onnx.py OUT_DIR [--chunk-mb 24] [--reference REF_DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange
from demucs.pretrained import get_model

MODEL_NAME = "htdemucs_6s"


class Core(torch.nn.Module):
    """HTDemucs.forward with the STFT and inverse STFT moved outside."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--chunk-mb", type=int, default=24)
    ap.add_argument("--reference", help="also write reference audio for tests here")
    ap.add_argument("--skip-export", action="store_true", help="only write the reference")
    args = ap.parse_args()

    bag = get_model(MODEL_NAME)
    model = bag.models[0]
    model.eval()
    assert model.cac and model.use_train_segment
    seg = int(model.segment * model.samplerate)
    core = Core(model).eval()

    torch.manual_seed(0)
    mix = torch.randn(1, 2, seg) * 0.1
    with torch.no_grad():
        z = model._spec(mix)
        mag = model._magnitude(z)
        xs, xt = core(mix, mag)
        # Check the split-out model against the untouched original.
        ref = model(mix)
        mine = model._ispec(model._mask(z, xs), seg) + xt
        err = (ref - mine).abs().max().item()
        print("wrapper vs original max abs err:", err)
        assert err < 1e-4

    if not args.skip_export:
        export(core, model, mix, mag, xs, xt, args)
    if args.reference:
        write_reference(bag, model, seg, args.reference)


def export(core, model, mix, mag, xs, xt, args):
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "model.onnx")
    torch.onnx.export(
        core, (mix, mag), onnx_path,
        input_names=["mix", "mag"], output_names=["spec", "wave"],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o_spec, o_wave = sess.run(None, {"mix": mix.numpy(), "mag": mag.numpy()})
    print("onnx vs torch spec err:", np.abs(o_spec - xs.numpy()).max(),
          "wave err:", np.abs(o_wave - xt.numpy()).max())

    # Split into chunks small enough for git hosting.
    data = open(onnx_path, "rb").read()
    step = args.chunk_mb * 1024 * 1024
    parts = []
    for i in range(0, len(data), step):
        name = f"model.onnx.part{len(parts):02d}"
        with open(os.path.join(args.out_dir, name), "wb") as f:
            f.write(data[i:i + step])
        parts.append(name)
    os.remove(onnx_path)
    manifest = {
        "name": MODEL_NAME, "sources": list(model.sources),
        "samplerate": model.samplerate,
        "segment": int(model.segment * model.samplerate),
        "nfft": model.nfft, "hop": model.hop_length,
        "bytes": len(data), "parts": parts,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(manifest)


def write_reference(bag, model, seg, out_dir):
    """A clip spanning several segments, separated by stock Demucs the same
    way `demucs.separate` does it (loudness-normalised, no random shifts)."""
    import soundfile as sf
    from demucs.apply import apply_model
    os.makedirs(out_dir, exist_ok=True)
    n = seg + seg // 2
    t = torch.arange(n) / model.samplerate
    torch.manual_seed(1)
    clip = torch.stack([
        0.3 * torch.sin(2 * torch.pi * 110 * t) + 0.05 * torch.randn(n),
        0.2 * torch.sin(2 * torch.pi * 440 * t) * (torch.sin(2 * torch.pi * 2 * t) > 0)
        + 0.05 * torch.randn(n),
    ])
    ref = clip.mean(0)
    with torch.no_grad():
        out = apply_model(bag, ((clip - ref.mean()) / ref.std())[None],
                          shifts=0, split=True, overlap=0.25)[0]
    out = out * ref.std() + ref.mean()
    sf.write(os.path.join(out_dir, "clip.wav"), clip.t().numpy(), model.samplerate,
             subtype="FLOAT")
    clip.numpy().astype("<f4").tofile(os.path.join(out_dir, "clip.f32"))
    out.numpy().astype("<f4").tofile(os.path.join(out_dir, "stems.f32"))
    print("reference written", tuple(clip.shape), tuple(out.shape))


if __name__ == "__main__":
    sys.exit(main())
