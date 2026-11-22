# Splitter

Browser-based audio stem splitter. Drop in an MP3 and get separate tracks for
**vocals, drums, bass, guitar, piano** and **other** (synths, strings, FX).
Everything runs on the device in the browser; no audio is uploaded anywhere.

It runs the open-source [Demucs](https://github.com/facebookresearch/demucs)
`htdemucs_6s` model (MIT) through ONNX Runtime Web.

## Run it

`web/` is a fully static site with no build step. Serve it over http(s):

    python tools/serve.py        # then open http://localhost:8000

Opening `index.html` straight from disk will not work.

## Notes

- The model (~115 MB, in `web/model/`, split into parts) downloads on first
  use and is cached by the browser afterwards.
- CPU mode takes roughly 1.5x the song's length on a 4-core desktop. The
  optional GPU (WebGPU) mode is experimental.
- A finished song holds 6 uncompressed stereo stems in memory (~130 MB per
  minute of audio), so long songs need a desktop-class device.
- `coi-serviceworker.js` enables multi-threading on hosts that cannot send
  COOP/COEP headers (e.g. GitHub Pages). It reloads the page once on first visit.

## Layout

- `web/dsp.js` – STFT / inverse STFT matching Demucs' conventions
- `web/separator.js` – chunking, normalisation and overlap-add
- `web/worker.js` – model download and inference
- `web/app.js` – UI, mixer playback, WAV/ZIP export
- `tools/export_onnx.py` – regenerates `web/model/` from the Demucs weights
- `tests/` – compare the JS pipeline (Node and headless Chromium) against
  stock Demucs output; usage is at the top of each file
