// WAV encoding and a minimal (uncompressed) ZIP writer for downloads.

export function encodeWav(channels, sampleRate) {
  const n = channels[0].length;
  const nch = channels.length;
  const header = new DataView(new ArrayBuffer(44));
  const dataBytes = n * nch * 2;
  const text = (o, s) => { for (let i = 0; i < s.length; i++) header.setUint8(o + i, s.charCodeAt(i)); };
  text(0, 'RIFF'); header.setUint32(4, 36 + dataBytes, true); text(8, 'WAVE');
  text(12, 'fmt '); header.setUint32(16, 16, true); header.setUint16(20, 1, true);
  header.setUint16(22, nch, true); header.setUint32(24, sampleRate, true);
  header.setUint32(28, sampleRate * nch * 2, true); header.setUint16(32, nch * 2, true);
  header.setUint16(34, 16, true); text(36, 'data'); header.setUint32(40, dataBytes, true);
  const pcm = new Int16Array(n * nch);
  for (let c = 0; c < nch; c++) {
    const src = channels[c];
    for (let i = 0; i < n; i++) {
      const v = Math.max(-1, Math.min(1, src[i]));
      pcm[i * nch + c] = Math.round(v * 32767);
    }
  }
  return [new Uint8Array(header.buffer), new Uint8Array(pcm.buffer)];
}

let crcTable = null;
function crc32(parts) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[i] = c;
    }
  }
  let crc = 0xffffffff;
  for (const p of parts) {
    for (let i = 0; i < p.length; i++) crc = crcTable[(crc ^ p[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// files: [{name, parts: Uint8Array[]}] -> Blob (stored, no compression).
export function zipBlob(files) {
  const enc = new TextEncoder();
  const out = [];
  const central = [];
  let offset = 0;
  for (const file of files) {
    const name = enc.encode(file.name);
    const size = file.parts.reduce((a, p) => a + p.length, 0);
    const crc = crc32(file.parts);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); local.setUint16(8, 0, true);
    local.setUint32(10, 0x00210000, true); // 1980-01-01 00:00
    local.setUint32(14, crc, true); local.setUint32(18, size, true); local.setUint32(22, size, true);
    local.setUint16(26, name.length, true); local.setUint16(28, 0, true);
    out.push(new Uint8Array(local.buffer), name, ...file.parts);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true);
    cd.setUint16(8, 0x0800, true); cd.setUint16(10, 0, true); cd.setUint32(12, 0x00210000, true);
    cd.setUint32(16, crc, true); cd.setUint32(20, size, true); cd.setUint32(24, size, true);
    cd.setUint16(28, name.length, true); cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), name);
    offset += 30 + name.length + size;
  }
  const cdSize = central.reduce((a, p) => a + p.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true); end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  return new Blob([...out, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}
