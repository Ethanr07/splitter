// Runs the exported Demucs network over a whole track, following the same
// chunking scheme as the reference implementation (demucs.apply.apply_model
// with shifts=0, overlap=0.25, plus the input normalisation from
// demucs.separate). Independent of any particular ONNX runtime: the caller
// supplies `run(mix, mag) -> {spec, wave}`.

import { Stft } from './dsp.js';

export const OVERLAP = 0.25;

export function segmentOffsets(length, segment) {
  const stride = Math.floor((1 - OVERLAP) * segment);
  const offsets = [];
  for (let o = 0; o < length; o += stride) offsets.push(o);
  return offsets;
}

// Mean and standard deviation of the mono mix, used to normalise loudness.
export function mixStats(channels) {
  const n = channels[0].length;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = (channels[0][i] + channels[1][i]) / 2;
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / n;
  // Unbiased estimator, as torch.std.
  const std = Math.sqrt(Math.max(0, (sumSq - n * mean * mean) / Math.max(1, n - 1)));
  return { mean, std: std > 1e-8 ? std : 1 };
}

// Separates `channels` ([left, right] Float32Arrays at the model sample rate).
// Calls `onChunk({index, total, offset, length, stems})` for each segment,
// where stems[s][c] is a Float32Array of `length` samples starting at `offset`.
// Chunks overlap; combine them with `OverlapAdder`.
export async function separate(channels, cfg, run, onChunk) {
  const total = channels[0].length;
  const { segment, nfft, hop } = cfg;
  const nSources = cfg.sources.length;
  const stft = new Stft(nfft, hop);
  const frames = stft.frames(segment);
  const plane = stft.bins * frames;
  const { mean, std } = mixStats(channels);
  const offsets = segmentOffsets(total, segment);

  for (let index = 0; index < offsets.length; index++) {
    const offset = offsets[index];
    const length = Math.min(segment, total - offset);
    // Short chunks are centred in a full-length window, borrowing real audio
    // from either side where it exists and zeros elsewhere.
    const delta = segment - length;
    const start = offset - Math.floor(delta / 2);
    const mix = new Float32Array(2 * segment);
    for (let c = 0; c < 2; c++) {
      const from = Math.max(0, start);
      const to = Math.min(total, start + segment);
      const src = channels[c];
      const base = c * segment - start;
      for (let i = from; i < to; i++) mix[base + i] = (src[i] - mean) / std;
    }
    const mag = new Float32Array(4 * plane);
    for (let c = 0; c < 2; c++) {
      stft.forward(mix.subarray(c * segment, (c + 1) * segment), mag, c * 2 * plane);
    }

    const { spec, wave } = await run(mix, mag);

    const trim = Math.floor(delta / 2);
    const stems = [];
    for (let s = 0; s < nSources; s++) {
      const pair = [];
      for (let c = 0; c < 2; c++) {
        const k = s * 2 + c;
        const full = new Float32Array(wave.subarray(k * segment, (k + 1) * segment));
        stft.inverseAdd(spec, k * 2 * plane, full);
        const out = new Float32Array(length);
        for (let i = 0; i < length; i++) out[i] = full[trim + i] * std + mean;
        pair.push(out);
      }
      stems.push(pair);
    }
    await onChunk({ index, total: offsets.length, offset, length, stems });
  }
}

// Blends overlapping chunks with a triangular cross-fade.
export class OverlapAdder {
  // `targets[s][c]` are zero-filled Float32Arrays of the full track length.
  constructor(targets, segment) {
    this.targets = targets;
    this.length = targets[0][0].length;
    this.sumWeight = new Float32Array(this.length);
    const half = Math.floor(segment / 2);
    this.weight = new Float32Array(segment);
    for (let i = 0; i < half; i++) this.weight[i] = (i + 1) / half;
    for (let i = half; i < segment; i++) this.weight[i] = (segment - i) / half;
  }

  add({ offset, length, stems }) {
    const w = this.weight;
    for (let i = 0; i < length; i++) this.sumWeight[offset + i] += w[i];
    for (let s = 0; s < stems.length; s++) {
      for (let c = 0; c < 2; c++) {
        const src = stems[s][c];
        const dst = this.targets[s][c];
        for (let i = 0; i < length; i++) dst[offset + i] += src[i] * w[i];
      }
    }
  }

  finish() {
    const sw = this.sumWeight;
    for (const pair of this.targets) {
      for (const dst of pair) {
        for (let i = 0; i < dst.length; i++) dst[i] /= sw[i];
      }
    }
  }
}
