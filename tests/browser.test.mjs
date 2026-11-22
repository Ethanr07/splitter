// Drives the real page in headless Chromium and compares its stems with the
// stock-Demucs reference.
//   python tools/serve.py 8123 &
//   node tests/browser.test.mjs REF_DIR [http://localhost:8123] [--gpu] [--shot out.png]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const { chromium } = createRequire(path.resolve(process.cwd(), 'x.js'))('playwright');

const args = process.argv.slice(2);
const flag = (name) => { const i = args.indexOf(name); return i < 0 ? null : args.splice(i, name === '--gpu' ? 1 : 2)[1] ?? true; };
const gpu = flag('--gpu');
const shot = flag('--shot');
const [refDir, url = 'http://localhost:8123'] = args;

const browser = await chromium.launch({ args: gpu ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader'] : [] });
const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page ${m.type()}]`, m.text()); });
page.on('pageerror', (e) => console.log('[page error]', e.message));
await page.goto(url);
await page.waitForFunction(() => window.splitter && window.crossOriginIsolated, null, { timeout: 30_000 });
if (gpu) {
  await page.waitForSelector('#gpuOption:not([hidden])', { timeout: 10_000 });
  await page.check('#gpu');
}
const t0 = Date.now();
await page.setInputFiles('#file', path.join(refDir, 'clip.wav'));
let last = '';
for (;;) {
  const state = await page.evaluate(() => ({
    status: document.getElementById('status').textContent,
    error: document.getElementById('error').hidden ? '' : document.getElementById('error').textContent,
    done: !!window.splitter.track,
  }));
  if (state.error) { console.error('FAIL:', state.error); process.exit(1); }
  if (state.done) break;
  if (state.status !== last) console.log(' ', (last = state.status));
  await page.waitForTimeout(1000);
}
console.log(`finished in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const b = fs.readFileSync(path.join(refDir, 'stems.f32'));
const want = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const sources = JSON.parse(fs.readFileSync(new URL('../web/model/manifest.json', import.meta.url))).sources;
const n = want.length / sources.length / 2;
let worst = 0;
for (const [s, name] of sources.entries()) {
  const got = await page.evaluate((name) => {
    const stem = window.splitter.track.stems.find((x) => x.name === name);
    return { absent: stem.absent, data: [0, 1].map((c) => Array.from(stem.buffer.getChannelData(c))) };
  }, name);
  let err = 0;
  for (let c = 0; c < 2; c++) {
    if (got.data[c].length !== n) { console.error('FAIL: length mismatch'); process.exit(1); }
    for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(got.data[c][i] - want[(s * 2 + c) * n + i]));
  }
  worst = Math.max(worst, err);
  console.log(`${name.padEnd(7)} max abs error ${err.toExponential(2)}${got.absent ? '  (flagged not detected)' : ''}`);
}
if (shot) await page.screenshot({ path: shot, fullPage: true });
await browser.close();
if (worst > 2e-3) { console.error('FAIL: output differs from reference Demucs'); process.exit(1); }
console.log('PASS');
