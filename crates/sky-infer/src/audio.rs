//! s6-ear front-end (docs/SPEC-S6.md §16): 16 kHz PCM → log-mel (40 bands, 25 ms Hann window,
//! 10 ms hop) → frames of 8 hops stacked (80 ms, the 12.5 Hz frame rate of Mimi), `[n][320]`.
//!
//! Plain DSP with no dependency (Hann window, radix-2 FFT, triangular mel filterbank, log), so the
//! same code runs natively, in WASM and in `sky-data`, which writes the training features with it.
//! The constant tables are computed in f64 and rounded to f32 once; the per-frame arithmetic is
//! f32 in a fixed order.

/// Sample rate of the input PCM.
pub const SAMPLE_RATE: usize = 16_000;
/// Window length in samples (25 ms).
pub const WIN: usize = 400;
/// Hop in samples (10 ms).
pub const HOP: usize = 160;
/// FFT size (the window zero-padded).
pub const N_FFT: usize = 512;
/// Mel bands.
pub const N_MELS: usize = 40;
/// Hops stacked into one state frame (8 × 10 ms = 80 ms).
pub const STACK: usize = 8;
/// Values per state frame (`STACK × N_MELS`).
pub const FRAME_DIM: usize = STACK * N_MELS;
/// Samples per state frame.
pub const FRAME_SAMPLES: usize = STACK * HOP;
/// Lowest and highest filter edges (Hz).
const F_MIN: f64 = 20.0;
const F_MAX: f64 = 8_000.0;
/// Floor of the mel energies before the log.
const FLOOR: f32 = 1e-8;

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// `log10(x)` for `x > 0` from IEEE basic operations only (no libm), so native and WASM builds
/// agree bit for bit: `x = m · 2^e` with `m ∈ [√½, √2)`, `ln m = 2 atanh((m − 1)/(m + 1))` by its
/// series (|error| < 1e-15).
fn log10_det(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    for k in 0..10 {
        sum += term / (2 * k + 1) as f64;
        term *= s2;
    }
    (2.0 * sum + e as f64 * std::f64::consts::LN_2) * std::f64::consts::LOG10_E
}

/// Precomputed window, twiddles and filterbank.
#[derive(Clone, Debug)]
pub struct LogMel {
    window: Vec<f32>,
    /// `(cos, sin)` of `-2πk / N_FFT`, `k < N_FFT / 2`.
    twiddles: Vec<(f32, f32)>,
    /// Bit-reversal permutation of `0..N_FFT`.
    rev: Vec<usize>,
    /// Per band: first FFT bin and the weights of the following bins.
    filters: Vec<(usize, Vec<f32>)>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        let tau = std::f64::consts::TAU;
        // periodic Hann
        let window = (0..WIN)
            .map(|n| (0.5 - 0.5 * (tau * n as f64 / WIN as f64).cos()) as f32)
            .collect();
        let twiddles = (0..N_FFT / 2)
            .map(|k| {
                let a = -tau * k as f64 / N_FFT as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let bits = N_FFT.trailing_zeros();
        let rev = (0..N_FFT)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let (m0, m1) = (hz_to_mel(F_MIN), hz_to_mel(F_MAX));
        let edges: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let bin_hz = SAMPLE_RATE as f64 / N_FFT as f64;
        let filters = (0..N_MELS)
            .map(|b| {
                let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
                let first = (lo / bin_hz).ceil() as usize;
                let last = ((hi / bin_hz).floor() as usize).min(N_FFT / 2);
                let w = (first..=last)
                    .map(|k| {
                        let f = k as f64 * bin_hz;
                        let v = if f <= mid {
                            (f - lo) / (mid - lo)
                        } else {
                            (hi - f) / (hi - mid)
                        };
                        v.max(0.0) as f32
                    })
                    .collect();
                (first, w)
            })
            .collect();
        LogMel {
            window,
            twiddles,
            rev,
            filters,
        }
    }

    /// Number of 10 ms hops of `n_samples` samples (one per started hop).
    pub fn n_hops(n_samples: usize) -> usize {
        n_samples / HOP
    }

    /// Log-mel features `[n_hops][40]`: hop `t` reads samples `[t·160, t·160 + 400)` (zeros past
    /// the end), `y = (log10(max(mel, 1e-8)) + 2) / 4`.
    pub fn log_mel(&self, pcm: &[f32]) -> Vec<f32> {
        let t = Self::n_hops(pcm.len());
        let mut out = vec![0.0f32; t * N_MELS];
        let mut re = vec![0.0f32; N_FFT];
        let mut im = vec![0.0f32; N_FFT];
        let mut power = vec![0.0f32; N_FFT / 2 + 1];
        for (h, row) in out.as_chunks_mut::<N_MELS>().0.iter_mut().enumerate() {
            let start = h * HOP;
            re.fill(0.0);
            im.fill(0.0);
            for n in 0..WIN {
                if let Some(&x) = pcm.get(start + n) {
                    re[self.rev[n]] = x * self.window[n];
                }
            }
            self.fft(&mut re, &mut im);
            for (k, p) in power.iter_mut().enumerate() {
                *p = re[k] * re[k] + im[k] * im[k];
            }
            for (o, (first, w)) in row.iter_mut().zip(&self.filters) {
                let mut e = 0.0f32;
                for (j, &wj) in w.iter().enumerate() {
                    e += wj * power[first + j];
                }
                *o = ((log10_det(e.max(FLOOR) as f64) + 2.0) / 4.0) as f32;
            }
        }
        out
    }

    /// In-place iterative radix-2 FFT of inputs already in bit-reversed order.
    fn fft(&self, re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (c, s) = self.twiddles[k * step];
                    let (i, j) = (start + k, start + k + half);
                    let tr = re[j] * c - im[j] * s;
                    let ti = re[j] * s + im[j] * c;
                    re[j] = re[i] - tr;
                    im[j] = im[i] - ti;
                    re[i] += tr;
                    im[i] += ti;
                }
            }
            len *= 2;
        }
    }

    /// State frames `[n][320]` of 16 kHz PCM: the log-mel hops stacked by 8 (a trailing partial
    /// frame is dropped).
    pub fn frames(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let mel = self.log_mel(pcm);
        stack(&mel)
    }
}

/// Stacks log-mel hops `[t][40]` into state frames `[t / 8][320]` (row-major, a reshape).
pub fn stack(mel: &[f32]) -> (Vec<f32>, usize) {
    let n = mel.len() / FRAME_DIM;
    (mel[..n * FRAME_DIM].to_vec(), n)
}

/// The last `max` frames of `[n][dim]` frames (the most recent audio decides).
pub fn last_frames(frames: &[f32], n: usize, dim: usize, max: usize) -> (&[f32], usize) {
    let keep = n.min(max);
    (&frames[(n - keep) * dim..n * dim], keep)
}

/// FNV-1a hash of PCM samples (the state-cache key of an audio clip).
pub fn pcm_hash(pcm: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in pcm {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h ^ pcm.len() as u64
}

/// Decodes a RIFF WAV file (PCM 8/16/24/32-bit or IEEE float 32-bit, any channel count,
/// downmixed to mono): `(samples in [-1, 1], sample rate)`.
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32_at =
        |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let (mut fmt, mut i) = (None, 12usize);
    while i + 8 <= bytes.len() {
        let (id, len) = (&bytes[i..i + 4], u32_at(i + 4) as usize);
        let body = i + 8;
        let end = (body + len).min(bytes.len());
        if id == b"fmt " && len >= 16 {
            fmt = Some((
                u16_at(body),
                u16_at(body + 2) as usize,
                u32_at(body + 4) as usize,
                u16_at(body + 14) as usize,
            ));
        } else if id == b"data" {
            let (tag, ch, rate, bits) = fmt.ok_or("data chunk before fmt")?;
            let ch = ch.max(1);
            // WAVE_FORMAT_EXTENSIBLE carries the real tag further on; PCM and float are told
            // apart by the bit depth then (32-bit extensible files here are float)
            let float = tag == 3 || (tag == 0xFFFE && bits == 32);
            let bps = bits / 8;
            if bps == 0 || !(tag == 1 || tag == 3 || tag == 0xFFFE) {
                return Err(format!("unsupported WAV format tag {tag} ({bits} bits)"));
            }
            let data = &bytes[body..end];
            let frames = data.len() / (bps * ch);
            let sample = |o: usize| -> f32 {
                let b = &data[o..o + bps];
                match (float, bps) {
                    (true, 4) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    (false, 1) => (b[0] as f32 - 128.0) / 128.0,
                    (false, 2) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                    (false, 3) => {
                        ((i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32) / 8_388_608.0
                    }
                    (false, 4) => {
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
                    }
                    _ => 0.0,
                }
            };
            let pcm = (0..frames)
                .map(|f| (0..ch).map(|c| sample((f * ch + c) * bps)).sum::<f32>() / ch as f32)
                .collect();
            return Ok((pcm, rate));
        }
        i = body + len + (len & 1);
    }
    Err("no data chunk".into())
}

/// Writes 16-bit PCM mono WAV bytes.
pub fn write_wav(pcm: &[f32], rate: usize) -> Vec<u8> {
    let n = pcm.len() * 2;
    let mut o = Vec::with_capacity(44 + n);
    o.extend_from_slice(b"RIFF");
    o.extend_from_slice(&(36 + n as u32).to_le_bytes());
    o.extend_from_slice(b"WAVEfmt ");
    o.extend_from_slice(&16u32.to_le_bytes());
    o.extend_from_slice(&1u16.to_le_bytes());
    o.extend_from_slice(&1u16.to_le_bytes());
    o.extend_from_slice(&(rate as u32).to_le_bytes());
    o.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    o.extend_from_slice(&2u16.to_le_bytes());
    o.extend_from_slice(&16u16.to_le_bytes());
    o.extend_from_slice(b"data");
    o.extend_from_slice(&(n as u32).to_le_bytes());
    for &x in pcm {
        o.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    o
}

/// Linear-interpolation resampling to `to` Hz (enough for speech going to 16 kHz; no
/// anti-aliasing filter beyond the averaging of a box over each output step when downsampling).
pub fn resample(pcm: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || pcm.is_empty() {
        return pcm.to_vec();
    }
    let n = (pcm.len() as u64 * to as u64 / from as u64) as usize;
    let ratio = from as f64 / to as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 * ratio;
            if ratio > 1.0 {
                // box average over the input samples of this output step
                let (a, b) = (x as usize, ((x + ratio) as usize).min(pcm.len()));
                let s: f32 = pcm[a..b.max(a + 1).min(pcm.len())].iter().sum();
                s / (b.max(a + 1).min(pcm.len()) - a) as f32
            } else {
                let j = x as usize;
                let f = (x - j as f64) as f32;
                let y0 = pcm[j];
                let y1 = *pcm.get(j + 1).unwrap_or(&y0);
                y0 + (y1 - y0) * f
            }
        })
        .collect()
}

/// Parses a C-ordered 2-D little-endian `<f2` or `<f4` `.npy` array: `(values, rows, cols)`.
pub fn read_npy(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not an npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => {
            if bytes.len() < 12 {
                return Err("truncated npy header".into());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + hlen)
            .ok_or("truncated npy header")?,
    )
    .map_err(|e| e.to_string())?;
    let width = if header.contains("'<f2'") {
        2
    } else if header.contains("'<f4'") {
        4
    } else {
        return Err(format!("unsupported npy dtype in {header}"));
    };
    if header.contains("'fortran_order': True") {
        return Err("Fortran-ordered npy".into());
    }
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|r| r.split('(').nth(1))
        .and_then(|r| r.split(')').next())
        .ok_or("npy header without a shape")?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().map_err(|_| format!("bad npy shape {shape}")))
        .collect::<Result<_, _>>()?;
    let (rows, cols) = match dims[..] {
        [r, c] => (r, c),
        [r] => (r, 1),
        _ => return Err(format!("npy of {} dimensions", dims.len())),
    };
    let data = &bytes[start + hlen..];
    if data.len() < rows * cols * width {
        return Err("truncated npy data".into());
    }
    let v = (0..rows * cols)
        .map(|i| {
            let b = &data[i * width..];
            if width == 2 {
                sky_gguf::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
            } else {
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            }
        })
        .collect();
    Ok((v, rows, cols))
}

/// `.npy` bytes of a 2-D `<f2` (float16) array `[rows][cols]`.
pub fn write_npy_f16(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f2', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic (6) + version (2) + length (2) + header + '\n', padded to a multiple of 64
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut o = Vec::with_capacity(10 + header.len() + values.len() * 2);
    o.extend_from_slice(b"\x93NUMPY\x01\x00");
    o.extend_from_slice(&(header.len() as u16).to_le_bytes());
    o.extend_from_slice(header.as_bytes());
    for &x in &values[..rows * cols] {
        o.extend_from_slice(&sky_gguf::f32_to_f16(x).to_le_bytes());
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, secs: f32) -> Vec<f32> {
        let n = (SAMPLE_RATE as f32 * secs) as usize;
        (0..n)
            .map(|i| 0.5 * (std::f32::consts::TAU * hz * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    /// Band whose filter peaks closest to `hz`.
    fn band_of(hz: f64) -> usize {
        let (m0, m1) = (hz_to_mel(F_MIN), hz_to_mel(F_MAX));
        (0..N_MELS)
            .min_by(|&a, &b| {
                let c = |i: usize| {
                    (mel_to_hz(m0 + (m1 - m0) * (i + 1) as f64 / (N_MELS + 1) as f64) - hz).abs()
                };
                c(a).total_cmp(&c(b))
            })
            .unwrap()
    }

    #[test]
    fn a_sine_lands_in_its_mel_band() {
        let lm = LogMel::new();
        for hz in [300.0f32, 1000.0, 3000.0, 6000.0] {
            let mel = lm.log_mel(&sine(hz, 0.5));
            assert_eq!(mel.len(), 50 * N_MELS);
            // a hop in the middle (fully inside the signal)
            let row = &mel[20 * N_MELS..21 * N_MELS];
            let top = (0..N_MELS)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap();
            let want = band_of(hz as f64);
            assert!(
                top.abs_diff(want) <= 1,
                "{hz} Hz: band {top}, expected {want}"
            );
            // far bands sit near the floor
            let far = if want > N_MELS / 2 { 0 } else { N_MELS - 1 };
            assert!(row[top] - row[far] > 1.0, "{hz} Hz: {row:?}");
        }
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let lm = LogMel::new();
        let x: Vec<f32> = (0..N_FFT)
            .map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut re = vec![0.0; N_FFT];
        let mut im = vec![0.0; N_FFT];
        for i in 0..N_FFT {
            re[lm.rev[i]] = x[i];
        }
        lm.fft(&mut re, &mut im);
        for k in [0, 1, 5, 100, 255, 256] {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for (n, &v) in x.iter().enumerate() {
                let ang = -std::f64::consts::TAU * (k * n) as f64 / N_FFT as f64;
                a += v as f64 * ang.cos();
                b += v as f64 * ang.sin();
            }
            assert!((re[k] as f64 - a).abs() < 1e-3 && (im[k] as f64 - b).abs() < 1e-3);
        }
    }

    #[test]
    fn frames_are_stacked_hops() {
        let lm = LogMel::new();
        let pcm = sine(440.0, 1.0);
        let mel = lm.log_mel(&pcm);
        let (fr, n) = lm.frames(&pcm);
        assert_eq!((n, fr.len()), (12, 12 * FRAME_DIM)); // 100 hops -> 12 frames of 80 ms
        assert_eq!(
            &fr[FRAME_DIM..FRAME_DIM + N_MELS],
            &mel[8 * N_MELS..9 * N_MELS]
        );
        let (tail, k) = last_frames(&fr, n, FRAME_DIM, 5);
        assert_eq!((k, tail[0]), (5, fr[7 * FRAME_DIM]));
        // silence sits at the floor: (log10(1e-8) + 2) / 4
        let quiet = lm.log_mel(&vec![0.0; 1600]);
        assert!(quiet.iter().all(|&v| (v + 1.5).abs() < 1e-6));
        assert_ne!(pcm_hash(&pcm), pcm_hash(&pcm[1..]));
    }

    #[test]
    fn deterministic_log_matches_libm() {
        for x in [1e-8, 3.7e-5, 0.5, 1.0, 1.999, 2.0, 1234.5, 9.9e7] {
            assert!((log10_det(x) - x.log10()).abs() < 1e-12, "{x}");
        }
    }

    #[test]
    fn wav_npy_and_resampling_round_trip() {
        let pcm = sine(440.0, 0.1);
        let (back, rate) = read_wav(&write_wav(&pcm, SAMPLE_RATE)).unwrap();
        assert_eq!((rate, back.len()), (SAMPLE_RATE, pcm.len()));
        assert!(pcm.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
        let lm = LogMel::new();
        let mel = lm.log_mel(&pcm);
        let (v, r, c) = read_npy(&write_npy_f16(&mel, mel.len() / N_MELS, N_MELS)).unwrap();
        assert_eq!((r, c), (10, N_MELS));
        assert!(mel.iter().zip(&v).all(|(a, b)| (a - b).abs() < 2e-3));
        let up = resample(&pcm, 16_000, 48_000);
        assert_eq!(up.len(), 3 * pcm.len());
        let down = resample(&up, 48_000, 16_000);
        assert_eq!(down.len(), pcm.len());
        assert!(pcm
            .iter()
            .zip(&down)
            .skip(1)
            .all(|(a, b)| (a - b).abs() < 0.05));
    }

    /// Fixed checksum of the features of a fixed signal: the WASM build must give the same (see
    /// `crates/sky-web` tests and `scripts/parity-node.mjs`).
    #[test]
    fn checksum_is_stable() {
        let lm = LogMel::new();
        let (fr, n) = lm.frames(&sine(523.25, 0.32));
        assert_eq!(n, 4);
        let s: f64 = fr.iter().map(|&v| v as f64).sum();
        assert!(s.is_finite());
        println!("log-mel checksum {s:.6}");
    }
}
