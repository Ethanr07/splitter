// End-to-end check of the JavaScript pipeline against stock Demucs.
//   python tools/export_onnx.py web/model --skip-export --reference REF_DIR
//   node tests/separator.test.mjs REF_DIR
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { separate, OverlapAdder } from '../web/separator.js';

const require = createRequire(path.resolve(process.cwd(), 'x.js'));
const ort = require('onnxruntime-node');

const refDir = process.argv[2];
const modelDir = new URL('../web/model/', import.meta.url).pathname;
const cfg = JSON.parse(fs.readFileSync(modelDir + 'manifest.json'));
const model = Buffer.concat(cfg.parts.map((p) => fs.readFileSync(modelDir + p)));
const f32 = (name) => {
  const b = fs.readFileSync(path.join(refDir, name));
  return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
};
const clip = f32('clip.f32');
const want = f32('stems.f32');
const n = clip.length / 2;
const channels = [clip.subarray(0, n), clip.subarray(n)];

const session = await ort.InferenceSession.create(model);
const frames = Math.ceil(cfg.segment / cfg.hop);
const run = async (mix, mag) => {
  const out = await session.run({
    mix: new ort.Tensor('float32', mix, [1, 2, cfg.segment]),
    mag: new ort.Tensor('float32', mag, [1, 4, cfg.nfft / 2, frames]),
  });
  return { spec: out.spec.data, wave: out.wave.data };
};

const got = cfg.sources.map(() => [new Float32Array(n), new Float32Array(n)]);
const adder = new OverlapAdder(got, cfg.segment);
const t0 = Date.now();
await separate(channels, cfg, run, (chunk) => adder.add(chunk));
adder.finish();
console.log(`separated ${(n / cfg.samplerate).toFixed(1)}s in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

let worst = 0;
cfg.sources.forEach((name, s) => {
  let err = 0, peak = 0;
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n; i++) {
      const w = want[(s * 2 + c) * n + i];
      err = Math.max(err, Math.abs(got[s][c][i] - w));
      peak = Math.max(peak, Math.abs(w));
    }
  }
  worst = Math.max(worst, err);
  console.log(`${name.padEnd(7)} max abs error ${err.toExponential(2)} (reference peak ${peak.toFixed(3)})`);
});
if (worst > 1e-3) {
  console.error('FAIL: output differs from reference Demucs');
  process.exit(1);
}
console.log('PASS');
