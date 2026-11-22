// Short-time Fourier transform helpers matching Demucs' conventions exactly
// (Hann window, nfft 4096, hop 1024, "normalized" scaling, reflect padding).
// Pure functions, shared by the browser worker and the Node tests.

export class FFT {
  constructor(n) {
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  // In-place complex FFT. `inverse` is unscaled (caller divides by n).
  transform(re, im, inverse) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cos[t];
          const wi = sign * sin[t];
          const a = start + k;
          const b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}

function reflectPad(x, left, right) {
  const n = x.length;
  const out = new Float32Array(n + left + right);
  out.set(x, left);
  for (let i = 0; i < left; i++) out[left - 1 - i] = x[i + 1];
  for (let i = 0; i < right; i++) out[left + n + i] = x[n - 2 - i];
  return out;
}

export class Stft {
  constructor(nfft, hop) {
    this.nfft = nfft;
    this.hop = hop;
    this.bins = nfft / 2; // Demucs drops the Nyquist bin.
    this.fft = new FFT(nfft);
    this.window = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft);
    this.re = new Float64Array(nfft);
    this.im = new Float64Array(nfft);
  }

  frames(length) {
    return Math.ceil(length / this.hop);
  }

  // Writes the spectrogram of one channel into `out`, laid out as two
  // consecutive [bins x frames] planes (real, then imaginary) from `outOffset`.
  forward(x, out, outOffset) {
    const { nfft, hop, bins, fft, window, re, im } = this;
    const le = this.frames(x.length);
    const pad = (hop / 2) * 3;
    let p = reflectPad(x, pad, pad + le * hop - x.length);
    p = reflectPad(p, nfft / 2, nfft / 2);
    const scale = 1 / Math.sqrt(nfft);
    const imOffset = outOffset + bins * le;
    for (let t = 0; t < le; t++) {
      const start = (t + 2) * hop;
      for (let j = 0; j < nfft; j++) {
        re[j] = p[start + j] * window[j];
        im[j] = 0;
      }
      fft.transform(re, im, false);
      for (let f = 0; f < bins; f++) {
        out[outOffset + f * le + t] = re[f] * scale;
        out[imOffset + f * le + t] = im[f] * scale;
      }
    }
  }

  // Inverse of `forward`: reads real/imag planes from `spec` at `specOffset`
  // and adds the reconstructed signal into `out` (length `out.length`).
  inverseAdd(spec, specOffset, out) {
    const { nfft, hop, bins, fft, window, re, im } = this;
    const length = out.length;
    const le = this.frames(length);
    const imOffset = specOffset + bins * le;
    const pad = (hop / 2) * 3;
    // sqrt(nfft) undoes the forward scaling, 1/nfft is the inverse FFT's,
    // and 1.5 is the sum of the squared, 4x-overlapped Hann windows.
    const scale = Math.sqrt(nfft) / nfft / 1.5;
    for (let t = 0; t < le; t++) {
      for (let f = 0; f < bins; f++) {
        re[f] = spec[specOffset + f * le + t];
        im[f] = spec[imOffset + f * le + t];
      }
      re[bins] = 0; im[bins] = 0;
      for (let f = 1; f < bins; f++) {
        re[nfft - f] = re[f];
        im[nfft - f] = -im[f];
      }
      fft.transform(re, im, true);
      const start = t * hop - pad;
      const j0 = Math.max(0, -start);
      const j1 = Math.min(nfft, length - start);
      for (let j = j0; j < j1; j++) out[start + j] += re[j] * window[j] * scale;
    }
  }
}
