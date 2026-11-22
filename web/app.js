import { OverlapAdder } from './separator.js';
import { encodeWav, zipBlob } from './export.js';

const SAMPLE_RATE = 44100;
const STEM_INFO = {
  vocals: { label: 'Vocals', color: '#e5484d' },
  drums: { label: 'Drums', color: '#f5a524' },
  bass: { label: 'Bass', color: '#3e63dd' },
  guitar: { label: 'Guitar', color: '#30a46c' },
  piano: { label: 'Piano', sub: 'keys', color: '#8e4ec6' },
  other: { label: 'Other', sub: 'synths, strings, FX', color: '#0091a7' },
};
const ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];
// A stem quieter than this, relative to the full mix, counts as absent.
const ABSENT_RATIO = 0.02;

const $ = (id) => document.getElementById(id);
const el = {
  drop: $('drop'), file: $('file'), gpu: $('gpu'), gpuOption: $('gpuOption'),
  progress: $('progress'), songName: $('songName'), cancel: $('cancel'), bar: $('bar'), status: $('status'),
  error: $('error'), result: $('result'), play: $('play'), time: $('time'), seek: $('seek'),
  zip: $('zip'), again: $('again'), stems: $('stems'),
};

let worker = null;
let job = null; // state of the separation in progress
let track = null; // finished result: { name, duration, stems: [{name, buffer, gain, ...}] }
const player = { ctx: null, sources: [], playing: false, offset: 0, startedAt: 0, raf: 0 };

const fmtTime = (s) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

function show(view) {
  el.drop.hidden = view !== 'drop';
  el.progress.hidden = view !== 'progress';
  el.result.hidden = view !== 'result';
}

function fail(message) {
  stopWorker();
  job = null;
  el.error.textContent = message;
  el.error.hidden = false;
  show('drop');
}

function stopWorker() {
  if (worker) worker.terminate();
  worker = null;
}

// ---------------------------------------------------------------- input ----

async function decode(file) {
  const data = await file.arrayBuffer();
  // An offline context at the model's rate makes the browser resample for us.
  const ctx = new OfflineAudioContext(2, 1, SAMPLE_RATE);
  const audio = await ctx.decodeAudioData(data);
  const left = audio.getChannelData(0);
  const right = audio.numberOfChannels > 1 ? audio.getChannelData(1) : left;
  return [new Float32Array(left), new Float32Array(right)];
}

async function start(file) {
  if (job) return;
  el.error.hidden = true;
  stopPlayback();
  track = null;
  job = { file, started: 0, firstChunkAt: 0 };
  el.songName.textContent = file.name;
  el.bar.removeAttribute('value');
  el.status.textContent = 'Reading audio…';
  show('progress');

  let channels;
  try {
    channels = await decode(file);
  } catch {
    return fail(`Couldn't read "${file.name}" as audio. Try an MP3 or WAV file.`);
  }
  if (!job) return; // cancelled while decoding
  const length = channels[0].length;
  if (length < SAMPLE_RATE / 2) return fail('That file is too short to split.');

  job.length = length;
  job.mix = channels.map((c) => c.slice()); // kept for presence detection
  job.backend = el.gpu.checked ? 'webgpu' : 'wasm';
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => onWorkerMessage(e.data);
  worker.onerror = (e) => fail(`Something went wrong while splitting: ${e.message || 'worker failed to start'}`);
  worker.postMessage({ type: 'separate', channels, backend: job.backend }, channels.map((c) => c.buffer));
}

function onWorkerMessage(msg) {
  if (!job) return;
  switch (msg.type) {
    case 'download':
      el.bar.value = msg.loaded / msg.total;
      el.status.textContent = `Downloading the separation model… ${(msg.loaded / 1e6).toFixed(0)} / ${(msg.total / 1e6).toFixed(0)} MB (first time only)`;
      break;
    case 'config': {
      job.cfg = msg.cfg;
      job.buffers = msg.cfg.sources.map(() => new AudioBuffer({ numberOfChannels: 2, length: job.length, sampleRate: SAMPLE_RATE }));
      job.data = job.buffers.map((b) => [b.getChannelData(0), b.getChannelData(1)]);
      job.adder = new OverlapAdder(job.data, msg.cfg.segment);
      break;
    }
    case 'stage':
      if (msg.stage === 'init') {
        el.bar.removeAttribute('value');
        el.status.textContent = 'Starting the separation engine…';
      } else if (msg.stage === 'separate') {
        job.started = performance.now();
        el.bar.value = 0;
        el.status.textContent = `Splitting on ${job.engine}…`;
      }
      break;
    case 'backend':
      job.engine = msg.backend === 'webgpu' ? 'GPU' : `CPU (${msg.threads} thread${msg.threads === 1 ? '' : 's'})`;
      break;
    case 'fallback':
      console.warn('GPU unavailable, using CPU:', msg.reason);
      break;
    case 'chunk': {
      const { chunk } = msg;
      job.adder.add(chunk);
      const done = chunk.index + 1;
      el.bar.value = done / chunk.total;
      const elapsed = (performance.now() - job.started) / 1000;
      const left = (elapsed / done) * (chunk.total - done);
      el.status.textContent = done < chunk.total
        ? `Splitting on ${job.engine}… ${Math.round((done / chunk.total) * 100)}% – about ${fmtTime(left)} left`
        : 'Finishing up…';
      break;
    }
    case 'done':
      finish();
      break;
    case 'error':
      fail(`Something went wrong while splitting: ${msg.message}`);
      break;
  }
}

function rms(channels) {
  let sum = 0;
  for (const c of channels) for (let i = 0; i < c.length; i += 4) sum += c[i] * c[i];
  return Math.sqrt(sum / (channels.length * Math.ceil(channels[0].length / 4)));
}

function finish() {
  job.adder.finish();
  // Chrome's AudioBuffer hands out live views; other browsers may copy, so
  // write the finished data back explicitly.
  job.buffers.forEach((b, s) => job.data[s].forEach((d, c) => b.copyToChannel(d, c)));
  const mixLevel = rms(job.mix) || 1;
  const stems = job.cfg.sources.map((name, s) => {
    const level = rms(job.data[s]);
    return { name, buffer: job.buffers[s], level, absent: level < mixLevel * ABSENT_RATIO, volume: 1, muted: false, solo: false };
  });
  stems.sort((a, b) => a.absent - b.absent || ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
  const base = job.file.name.replace(/\.[^.]+$/, '') || 'song';
  track = { name: base, duration: job.length / SAMPLE_RATE, stems };
  job = null;
  stopWorker();
  renderResult();
  show('result');
}

// --------------------------------------------------------------- output ----

function drawWave(canvas, buffer, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.round(56 * dpr);
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d');
  const data = buffer.getChannelData(0);
  const per = data.length / w;
  g.fillStyle = color;
  for (let x = 0; x < w; x++) {
    let peak = 0;
    const end = Math.min(data.length, Math.floor((x + 1) * per));
    const stride = Math.max(1, Math.floor(per / 64));
    for (let i = Math.floor(x * per); i < end; i += stride) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
    }
    const bar = Math.max(dpr, Math.min(1, peak) * h);
    g.fillRect(x, (h - bar) / 2, 1, bar);
  }
}

function save(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

const stemWav = (stem) => encodeWav([stem.buffer.getChannelData(0), stem.buffer.getChannelData(1)], SAMPLE_RATE);

function renderResult() {
  el.stems.textContent = '';
  for (const stem of track.stems) {
    const info = STEM_INFO[stem.name] || { label: stem.name, color: '#888' };
    const row = document.createElement('div');
    row.className = `card stem${stem.absent ? ' absent' : ''}`;
    row.dataset.stem = stem.name;

    const name = document.createElement('div');
    name.className = 'name';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = info.color;
    const text = document.createElement('span');
    text.textContent = info.label;
    const sub = document.createElement('span');
    sub.className = 'sub';
    sub.textContent = stem.absent ? 'not detected' : info.sub || '';
    text.append(sub);
    name.append(dot, text);

    const wrap = document.createElement('div');
    wrap.className = 'wave';
    wrap.style.position = 'relative';
    const canvas = document.createElement('canvas');
    const head = document.createElement('div');
    head.className = 'playhead';
    head.style.cssText = 'position:absolute;top:0;bottom:0;width:2px;background:var(--text);left:0;pointer-events:none';
    wrap.append(canvas, head);
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      seekTo(((e.clientX - r.left) / r.width) * track.duration);
    });
    stem.canvas = canvas;
    stem.head = head;
    stem.color = info.color;

    const controls = document.createElement('div');
    controls.className = 'controls';
    const mute = document.createElement('button');
    mute.type = 'button'; mute.textContent = 'M'; mute.title = 'Mute'; mute.setAttribute('aria-label', `Mute ${info.label}`);
    const solo = document.createElement('button');
    solo.type = 'button'; solo.textContent = 'S'; solo.title = 'Solo'; solo.setAttribute('aria-label', `Solo ${info.label}`);
    const vol = document.createElement('input');
    vol.type = 'range'; vol.min = 0; vol.max = 1.5; vol.step = 0.01; vol.value = 1; vol.setAttribute('aria-label', `${info.label} volume`);
    const dl = document.createElement('button');
    dl.type = 'button'; dl.textContent = 'WAV'; dl.title = `Download ${info.label} as WAV`;
    mute.addEventListener('click', () => { stem.muted = !stem.muted; mute.classList.toggle('on', stem.muted); applyGains(); });
    solo.addEventListener('click', () => { stem.solo = !stem.solo; solo.classList.toggle('on', stem.solo); applyGains(); });
    vol.addEventListener('input', () => { stem.volume = Number(vol.value); applyGains(); });
    dl.addEventListener('click', () => save(new Blob(stemWav(stem), { type: 'audio/wav' }), `${track.name} - ${stem.name}.wav`));
    controls.append(mute, solo, vol, dl);

    row.append(name, wrap, controls);
    el.stems.append(row);
  }
  requestAnimationFrame(() => track?.stems.forEach((s) => drawWave(s.canvas, s.buffer, s.color)));
  player.offset = 0;
  updateTransport();
}

// ------------------------------------------------------------- playback ----

function stemGain(stem) {
  const anySolo = track.stems.some((s) => s.solo);
  if (stem.muted || (anySolo && !stem.solo)) return 0;
  return stem.volume;
}

function applyGains() {
  if (!player.ctx || !track) return;
  for (const stem of track.stems) {
    if (stem.gain) stem.gain.gain.setTargetAtTime(stemGain(stem), player.ctx.currentTime, 0.01);
  }
}

function position() {
  if (!player.playing) return player.offset;
  return Math.min(track.duration, player.offset + player.ctx.currentTime - player.startedAt);
}

function startPlayback() {
  if (!track) return;
  if (!player.ctx) player.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  player.ctx.resume();
  if (player.offset >= track.duration - 0.05) player.offset = 0;
  const when = player.ctx.currentTime + 0.05;
  player.sources = track.stems.map((stem) => {
    const src = player.ctx.createBufferSource();
    src.buffer = stem.buffer;
    stem.gain = player.ctx.createGain();
    stem.gain.gain.value = stemGain(stem);
    src.connect(stem.gain).connect(player.ctx.destination);
    src.start(when, player.offset);
    return src;
  });
  player.startedAt = when;
  player.playing = true;
  tick();
}

function stopPlayback() {
  if (player.playing) player.offset = position();
  for (const src of player.sources) {
    try { src.stop(); } catch { /* already stopped */ }
    src.disconnect();
  }
  player.sources = [];
  player.playing = false;
  cancelAnimationFrame(player.raf);
  if (track) updateTransport();
}

function seekTo(seconds) {
  const wasPlaying = player.playing;
  stopPlayback();
  player.offset = Math.max(0, Math.min(track.duration, seconds));
  if (wasPlaying) startPlayback();
  else updateTransport();
}

function updateTransport() {
  const pos = position();
  const frac = track.duration ? pos / track.duration : 0;
  el.time.textContent = `${fmtTime(pos)} / ${fmtTime(track.duration)}`;
  if (document.activeElement !== el.seek || !seeking) el.seek.value = frac;
  for (const stem of track.stems) if (stem.head) stem.head.style.left = `${frac * 100}%`;
  el.play.textContent = player.playing ? '❚❚' : '▶';
  el.play.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
}

function tick() {
  if (!player.playing) return;
  if (position() >= track.duration) {
    stopPlayback();
    player.offset = 0;
    updateTransport();
    return;
  }
  updateTransport();
  player.raf = requestAnimationFrame(tick);
}

// --------------------------------------------------------------- wiring ----

let seeking = false;
el.seek.addEventListener('pointerdown', () => { seeking = true; });
el.seek.addEventListener('change', () => { seeking = false; seekTo(Number(el.seek.value) * track.duration); });
el.play.addEventListener('click', () => (player.playing ? stopPlayback() : startPlayback()));
el.cancel.addEventListener('click', () => { stopWorker(); job = null; show('drop'); });
el.again.addEventListener('click', () => { stopPlayback(); track = null; el.stems.textContent = ''; show('drop'); });
el.zip.addEventListener('click', () => {
  el.zip.disabled = true;
  el.zip.textContent = 'Preparing…';
  // Let the button repaint before the (blocking) encode.
  setTimeout(() => {
    try {
      const files = track.stems.filter((s) => !s.absent).map((s) => ({ name: `${track.name} - ${s.name}.wav`, parts: stemWav(s) }));
      save(zipBlob(files), `${track.name} - stems.zip`);
    } finally {
      el.zip.disabled = false;
      el.zip.textContent = 'Download all (ZIP)';
    }
  }, 30);
});

el.drop.addEventListener('click', (e) => { if (!e.target.closest('.option')) el.file.click(); });
el.drop.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && e.target === el.drop) { e.preventDefault(); el.file.click(); }
});
el.file.addEventListener('change', () => { const f = el.file.files[0]; el.file.value = ''; if (f) start(f); });
for (const type of ['dragenter', 'dragover']) {
  window.addEventListener(type, (e) => { e.preventDefault(); el.drop.classList.add('over'); });
}
for (const type of ['dragleave', 'drop']) {
  window.addEventListener(type, (e) => { e.preventDefault(); el.drop.classList.remove('over'); });
}
window.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && !job && !el.drop.hidden) start(f);
});
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && track && !/^(INPUT|BUTTON)$/.test(e.target.tagName)) {
    e.preventDefault();
    player.playing ? stopPlayback() : startPlayback();
  }
});
let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => track?.stems.forEach((s) => drawWave(s.canvas, s.buffer, s.color)), 150);
});

if (navigator.gpu) {
  navigator.gpu.requestAdapter().then((adapter) => { if (adapter) el.gpuOption.hidden = false; }).catch(() => {});
}
if (location.protocol === 'file:') {
  fail('This page needs to be served over http(s) – see the README for a one-line way to run it locally.');
}

// Exposed for automated tests.
window.splitter = { get track() { return track; }, get job() { return job; } };
