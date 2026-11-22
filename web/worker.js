// Inference worker: downloads the model, runs it with ONNX Runtime Web and
// streams separated chunks back to the page.

import { separate } from './separator.js';

const MODEL_DIR = new URL('./model/', import.meta.url);
const ORT_DIR = new URL('./vendor/ort/', import.meta.url);
const CACHE_NAME = 'splitter-model-v1';

let cfg = null;
let modelBytes = null;
let ort = null;
let session = null;
let backend = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function fetchCached(url) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return hit;
  } catch {
    cache = null; // Cache storage unavailable (private mode etc.); just fetch.
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url} (HTTP ${res.status})`);
  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch {
      // Storage quota exceeded: carry on without caching.
    }
  }
  return res;
}

async function loadModel() {
  if (modelBytes) return;
  cfg = await (await fetch(new URL('manifest.json', MODEL_DIR))).json();
  const bytes = new Uint8Array(cfg.bytes);
  let loaded = 0;
  for (const part of cfg.parts) {
    const res = await fetchCached(new URL(part, MODEL_DIR).href);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.length > bytes.length) throw new Error('Model download is larger than expected');
      bytes.set(value, loaded);
      loaded += value.length;
      post({ type: 'download', loaded, total: cfg.bytes });
    }
  }
  if (loaded !== cfg.bytes) throw new Error('Model download is incomplete');
  modelBytes = bytes;
}

async function createSession(want) {
  ort = await import(new URL(want === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs', ORT_DIR).href);
  ort.env.wasm.wasmPaths = ORT_DIR.href;
  // Multi-threading needs a cross-origin isolated page (see coi-serviceworker.js).
  const cores = navigator.hardwareConcurrency || 4;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(cores, 8)) : 1;
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [want],
    graphOptimizationLevel: 'all',
  });
  backend = want;
  post({ type: 'backend', backend, threads: want === 'wasm' ? ort.env.wasm.numThreads : 0 });
}

async function infer(mix, mag) {
  const frames = Math.ceil(cfg.segment / cfg.hop);
  const out = await session.run({
    mix: new ort.Tensor('float32', mix, [1, 2, cfg.segment]),
    mag: new ort.Tensor('float32', mag, [1, 4, cfg.nfft / 2, frames]),
  });
  const result = { spec: await out.spec.getData(), wave: await out.wave.getData() };
  out.spec.dispose?.();
  out.wave.dispose?.();
  for (const v of result.spec.subarray(0, 4096)) {
    if (!Number.isFinite(v)) throw new Error('Model produced invalid output');
  }
  return result;
}

async function run(channels, wantBackend) {
  post({ type: 'stage', stage: 'download' });
  await loadModel();
  post({ type: 'config', cfg });

  post({ type: 'stage', stage: 'init' });
  if (!session || backend !== wantBackend) {
    try {
      await createSession(wantBackend);
    } catch (err) {
      if (wantBackend !== 'webgpu') throw err;
      post({ type: 'fallback', reason: String(err?.message || err) });
      await createSession('wasm');
    }
  }

  post({ type: 'stage', stage: 'separate' });
  let verified = false;
  const guardedInfer = async (mix, mag) => {
    if (verified || backend !== 'webgpu') return infer(mix, mag);
    // If the GPU path fails on its first real run, retry on the CPU.
    try {
      const out = await infer(mix, mag);
      verified = true;
      return out;
    } catch (err) {
      post({ type: 'fallback', reason: String(err?.message || err) });
      await createSession('wasm');
      return infer(mix, mag);
    }
  };
  await separate(channels, cfg, guardedInfer, (chunk) => {
    const transfer = chunk.stems.flat().map((a) => a.buffer);
    post({ type: 'chunk', chunk }, transfer);
  });
  post({ type: 'done' });
}

self.onmessage = (e) => {
  const { type, channels, backend: wantBackend } = e.data;
  if (type !== 'separate') return;
  run(channels, wantBackend).catch((err) => {
    post({ type: 'error', message: String(err?.message || err) });
  });
};
