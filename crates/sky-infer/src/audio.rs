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

/// An in-place iterative radix-2 FFT of a fixed power-of-two size (twiddles computed in f64 and
/// rounded to f32 once; the butterflies are f32 in a fixed order).
#[derive(Clone, Debug)]
pub struct Fft {
    /// `(cos, sin)` of `-2πk / n`, `k < n / 2`.
    twiddles: Vec<(f32, f32)>,
    /// Bit-reversal permutation of `0..n`.
    rev: Vec<usize>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "FFT size {n}");
        let tau = std::f64::consts::TAU;
        let twiddles = (0..n / 2)
            .map(|k| {
                let a = -tau * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        Fft { twiddles, rev }
    }

    /// Index in the bit-reversed input order of input sample `i`.
    #[inline]
    pub fn slot(&self, i: usize) -> usize {
        self.rev[i]
    }

    /// In-place FFT of inputs already in bit-reversed order ([`Fft::slot`]).
    pub fn run(&self, re: &mut [f32], im: &mut [f32]) {
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
}

/// Precomputed window, twiddles and filterbank.
#[derive(Clone, Debug)]
pub struct LogMel {
    window: Vec<f32>,
    fft: Fft,
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
            fft: Fft::new(N_FFT),
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
        let mut energies = [0.0f32; N_MELS];
        for (h, row) in out.as_chunks_mut::<N_MELS>().0.iter_mut().enumerate() {
            let start = h * HOP;
            let end = (start + WIN).min(pcm.len());
            self.mel_energies(
                &pcm[start.min(end)..end],
                &mut re,
                &mut im,
                &mut power,
                &mut energies,
            );
            for (o, &e) in row.iter_mut().zip(&energies) {
                *o = ((log10_det(e.max(FLOOR) as f64) + 2.0) / 4.0) as f32;
            }
        }
        out
    }

    /// The 40 mel energies (before the log) of one window of samples (`frame`, at most `WIN`
    /// samples, zeros past its end), with scratch buffers `re`, `im` (`N_FFT`) and `power`
    /// (`N_FFT / 2 + 1`).
    fn mel_energies(
        &self,
        frame: &[f32],
        re: &mut [f32],
        im: &mut [f32],
        power: &mut [f32],
        out: &mut [f32; N_MELS],
    ) {
        re.fill(0.0);
        im.fill(0.0);
        for (n, &x) in frame.iter().take(WIN).enumerate() {
            re[self.fft.slot(n)] = x * self.window[n];
        }
        self.fft.run(re, im);
        for (k, p) in power.iter_mut().enumerate() {
            *p = re[k] * re[k] + im[k] * im[k];
        }
        for (o, (first, w)) in out.iter_mut().zip(&self.filters) {
            let mut e = 0.0f32;
            for (j, &wj) in w.iter().enumerate() {
                e += wj * power[first + j];
            }
            *o = e;
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

// ----------------------------------------------------------------------------- s6-ear v2 (turn)

/// s6-ear v2 (docs/s6-ear2-prereg.md §4.1): one 10 ms hop reads a span of 40 ms of samples, the
/// analysis window of the prosody channels; the 25 ms log-mel window sits in its middle. A hop
/// exists once its whole span has arrived, so a stream and a batch computation give the same hops
/// (the grid starts at the first sample).
pub const TURN_SPAN: usize = 640;
/// FFT size of the prosody analysis (the 40 ms window zero-padded: lags up to 384 alias-free).
pub const PROSODY_FFT: usize = 1024;
/// Lag search of the pitch (samples): 80–400 Hz.
pub const LAG_MIN: usize = 40;
pub const LAG_MAX: usize = 200;
/// Voicing from which a hop carries its pitch.
pub const VOICED: f32 = 0.45;
/// Praat's octave cost (per octave of lag): the tie-break that keeps the shortest of the equal
/// lags of a periodic signal (its period, not a multiple of it).
pub const OCTAVE_COST: f64 = 0.01;
/// Channels of a hop: 40 log10 mel values, then log2(f0 / 100 Hz), voicing, log10 energy.
pub const TURN_CHANNELS: usize = N_MELS + 3;
/// Index of the f0, voicing and energy channels in a hop.
pub const CH_F0: usize = N_MELS;
pub const CH_VOICING: usize = N_MELS + 1;
pub const CH_ENERGY: usize = N_MELS + 2;
/// Hops of a decision window (4.0 s).
pub const TURN_WINDOW_HOPS: usize = 400;
/// Offset of the log-mel window inside a hop's span (centred).
const MEL_OFFSET: usize = (TURN_SPAN - WIN) / 2;
/// `log2(10)`, to take `log2` from [`log10_det`].
const LOG2_10: f64 = std::f64::consts::LOG2_10;

/// Number of complete hops of `n_samples` samples.
pub fn turn_hop_count(n_samples: usize) -> usize {
    if n_samples < TURN_SPAN {
        0
    } else {
        (n_samples - TURN_SPAN) / HOP + 1
    }
}

/// The pitch analysis of one hop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prosody {
    /// The normalized autocorrelation at the chosen peak, in `[0, 1]`.
    pub voicing: f32,
    /// The pitch in Hz at that peak (parabolic interpolation), 0 without a peak.
    pub f0: f32,
}

/// The s6-ear v2 front end: per 10 ms hop, the 40 log10 mel values of v1's filterbank (stored
/// before any normalisation) and the three prosody channels of [`TurnFrontEnd::prosody`].
#[derive(Clone, Debug)]
pub struct TurnFrontEnd {
    lm: LogMel,
    /// 40 ms periodic Hann window.
    pwin: Vec<f32>,
    pfft: Fft,
    /// The window's own normalized autocorrelation `r_w(lag) / r_w(0)`, `lag ≤ LAG_MAX + 1`.
    rw: Vec<f64>,
    re: Vec<f32>,
    im: Vec<f32>,
    power: Vec<f32>,
    pre: Vec<f32>,
    pim: Vec<f32>,
    pspec: Vec<f32>,
}

impl Default for TurnFrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnFrontEnd {
    pub fn new() -> Self {
        let tau = std::f64::consts::TAU;
        let pwin: Vec<f32> = (0..TURN_SPAN)
            .map(|n| (0.5 - 0.5 * (tau * n as f64 / TURN_SPAN as f64).cos()) as f32)
            .collect();
        // the window's autocorrelation, directly in f64 (once)
        let r = |lag: usize| -> f64 {
            (0..TURN_SPAN - lag)
                .map(|n| pwin[n] as f64 * pwin[n + lag] as f64)
                .sum()
        };
        let r0 = r(0);
        let rw = (0..=LAG_MAX + 1).map(|l| r(l) / r0).collect();
        TurnFrontEnd {
            lm: LogMel::new(),
            pwin,
            pfft: Fft::new(PROSODY_FFT),
            rw,
            re: vec![0.0; N_FFT],
            im: vec![0.0; N_FFT],
            power: vec![0.0; N_FFT / 2 + 1],
            pre: vec![0.0; PROSODY_FFT],
            pim: vec![0.0; PROSODY_FFT],
            pspec: vec![0.0; PROSODY_FFT],
        }
    }

    /// Voicing and pitch of one 40 ms span: the autocorrelation is the inverse FFT of the power
    /// spectrum of the Hann-windowed span (1,024 points), normalized by its value at lag 0 and by
    /// the window's own autocorrelation (Boersma 1993). Among the local maxima of lags 40–200,
    /// the one of highest `R + OCTAVE_COST · log2(LAG_MAX / lag)` is kept, its lag and value
    /// refined by a parabola through its neighbours.
    pub fn prosody(&mut self, span: &[f32]) -> Prosody {
        let (re, im, spec) = (&mut self.pre, &mut self.pim, &mut self.pspec);
        re.fill(0.0);
        im.fill(0.0);
        for (n, &x) in span.iter().take(TURN_SPAN).enumerate() {
            re[self.pfft.slot(n)] = x * self.pwin[n];
        }
        self.pfft.run(re, im);
        // the power spectrum is real and even: its forward FFT is its inverse FFT times the size
        for (k, p) in spec.iter_mut().enumerate() {
            *p = re[k] * re[k] + im[k] * im[k];
        }
        re.fill(0.0);
        im.fill(0.0);
        for (k, &p) in spec.iter().enumerate() {
            re[self.pfft.slot(k)] = p;
        }
        self.pfft.run(re, im);
        let none = Prosody {
            voicing: 0.0,
            f0: 0.0,
        };
        let r0 = re[0] as f64;
        if r0 <= 1e-20 {
            return none;
        }
        let rw = &self.rw;
        let norm = |lag: usize| re[lag] as f64 / r0 / rw[lag];
        let mut best: Option<(f64, f64, f64)> = None; // (score, lag, value)
        for lag in LAG_MIN..=LAG_MAX {
            let (a, b, c) = (norm(lag - 1), norm(lag), norm(lag + 1));
            if !(b >= a && b > c) {
                continue;
            }
            let den = a - 2.0 * b + c;
            let (shift, value) = if den < 0.0 {
                let d = 0.5 * (a - c) / den;
                (d, b - 0.25 * (a - c) * d)
            } else {
                (0.0, b)
            };
            let l = lag as f64 + shift;
            let score = value + OCTAVE_COST * log10_det(LAG_MAX as f64 / l) * LOG2_10;
            if best.is_none_or(|(s, _, _)| score > s) {
                best = Some((score, l, value));
            }
        }
        match best {
            None => none,
            Some((_, lag, value)) => Prosody {
                voicing: value.clamp(0.0, 1.0) as f32,
                f0: (SAMPLE_RATE as f64 / lag) as f32,
            },
        }
    }

    /// The 43 channels of one hop from its 40 ms span (zeros past the end of a shorter slice):
    /// the 40 log10 mel values of the centred 25 ms window, then log2(f0 / 100 Hz) when the
    /// voicing is at least [`VOICED`] (else 0), the voicing, and log10 of the sum of the 40 mel
    /// energies. Floors as v1 (`1e-8`); every log is [`log10_det`].
    pub fn hop(&mut self, span: &[f32], out: &mut [f32]) {
        let mut e = [0.0f32; N_MELS];
        let frame = span.get(MEL_OFFSET..).unwrap_or(&[]);
        self.lm
            .mel_energies(frame, &mut self.re, &mut self.im, &mut self.power, &mut e);
        let mut total = 0.0f32;
        for (o, &v) in out.iter_mut().zip(&e) {
            *o = log10_det(v.max(FLOOR) as f64) as f32;
            total += v;
        }
        let p = self.prosody(span);
        out[CH_F0] = if p.voicing >= VOICED && p.f0 > 0.0 {
            (log10_det(p.f0 as f64 / 100.0) * LOG2_10) as f32
        } else {
            0.0
        };
        out[CH_VOICING] = p.voicing;
        out[CH_ENERGY] = log10_det(total.max(FLOOR) as f64) as f32;
    }

    /// The raw hops `[n][43]` of 16 kHz PCM, at most its last `max` hops (0: all): hop `t` reads
    /// samples `[t·160, t·160 + 640)`.
    pub fn hops(&mut self, pcm: &[f32], max: usize) -> (Vec<f32>, usize) {
        let n = turn_hop_count(pcm.len());
        let first = if max > 0 { n.saturating_sub(max) } else { 0 };
        let mut out = vec![0.0f32; (n - first) * TURN_CHANNELS];
        for (i, row) in out
            .as_chunks_mut::<TURN_CHANNELS>()
            .0
            .iter_mut()
            .enumerate()
        {
            let s = (first + i) * HOP;
            self.hop(&pcm[s..s + TURN_SPAN], row);
        }
        (out, n - first)
    }
}

/// The model input of a decision window (docs/s6-ear2-prereg.md §4.1, "Normalisation" and
/// "Window and positions"): of the raw hops `[n][43]`, the last `positions · per_pos` hops with
/// `positions = n / per_pos` (the grid ends at the last hop), normalised over those hops by
/// [`normalize_hops`] and laid out as `[positions][per_pos · 43]` (hop-major). No padding: a
/// short window gives fewer positions.
pub fn normalize_window(hops: &[f32], n: usize, per_pos: usize) -> (Vec<f32>, usize) {
    let positions = n / per_pos;
    let keep = positions * per_pos;
    let mut out = hops[(n - keep) * TURN_CHANNELS..n * TURN_CHANNELS].to_vec();
    normalize_hops(&mut out);
    (out, positions)
}

/// Whisper's rule over raw hops `[n][43]` in place: with `M` their largest log10 mel value, each
/// mel value `x ← (max(x, M − 8) − M + 4) / 4`, the energy shifted the same way
/// `(x − M + 4) / 4`, f0 and voicing left as they are.
pub fn normalize_hops(rows: &mut [f32]) {
    let m = rows
        .as_chunks::<TURN_CHANNELS>()
        .0
        .iter()
        .flat_map(|r| r[..N_MELS].iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return;
    }
    let lo = m - 8.0;
    for r in rows.as_chunks_mut::<TURN_CHANNELS>().0.iter_mut() {
        for v in &mut r[..N_MELS] {
            *v = (v.max(lo) - m + 4.0) / 4.0;
        }
        r[CH_ENERGY] = (r[CH_ENERGY] - m + 4.0) / 4.0;
    }
}

/// A deterministic speech-like test signal (golden files, benchmarks, parity): a pulse train whose
/// pitch glides around 120 Hz, plus white noise, scaled by `amp` (basic operations and `sin`/`cos`
/// of f64 only; the same samples on every target up to libm's last bit).
pub fn synth_speech(secs: f64, seed: u64, amp: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f64 * secs) as usize;
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            let f0 = 120.0 + 40.0 * (t * 1.3 + seed as f64).sin();
            phase += f0 / SAMPLE_RATE as f64;
            let p = phase.fract();
            let pulse = if p < 0.08 {
                0.5 - 0.5 * (std::f64::consts::TAU * p / 0.08).cos()
            } else {
                0.0
            };
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            amp * (0.4 * pulse + 0.02 * noise) as f32
        })
        .collect()
}

/// The streaming front end (docs/s6-ear2-prereg.md §4.1, "Streaming"): accepts PCM in chunks of
/// any size, computes each hop once, keeps the last `max_hops` raw hops. They are the hops of
/// [`TurnFrontEnd::hops`] on the concatenated samples, bit for bit.
#[derive(Clone, Debug)]
pub struct FeatureStream {
    fe: TurnFrontEnd,
    max_hops: usize,
    /// Samples from the start of the next hop's span on.
    pending: Vec<f32>,
    /// Ring of the last hops, `[max_hops][43]`; `head` is the oldest slot.
    ring: Vec<f32>,
    head: usize,
    len: usize,
    /// Hops seen since the start (or the last reset).
    total: u64,
}

impl FeatureStream {
    pub fn new(max_hops: usize) -> Self {
        assert!(max_hops > 0, "a stream keeps at least one hop");
        FeatureStream {
            fe: TurnFrontEnd::new(),
            max_hops,
            pending: Vec::new(),
            ring: vec![0.0; max_hops * TURN_CHANNELS],
            head: 0,
            len: 0,
            total: 0,
        }
    }

    /// Forgets every sample and hop.
    pub fn reset(&mut self) {
        self.pending.clear();
        (self.head, self.len, self.total) = (0, 0, 0);
    }

    /// Appends 16 kHz samples and computes the hops they complete. Hops that the same call pushes
    /// out of the kept window are counted but not computed.
    pub fn push(&mut self, pcm: &[f32]) {
        self.pending.extend_from_slice(pcm);
        let new = turn_hop_count(self.pending.len());
        let skip = new.saturating_sub(self.max_hops);
        self.total += skip as u64;
        let mut row = [0.0f32; TURN_CHANNELS];
        for i in skip..new {
            let s = i * HOP;
            self.fe.hop(&self.pending[s..s + TURN_SPAN], &mut row);
            self.put(&row);
        }
        self.pending.drain(..new * HOP);
    }

    fn put(&mut self, row: &[f32]) {
        let slot = (self.head + self.len) % self.max_hops;
        self.ring[slot * TURN_CHANNELS..(slot + 1) * TURN_CHANNELS].copy_from_slice(row);
        if self.len < self.max_hops {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % self.max_hops;
        }
        self.total += 1;
    }

    /// Hops kept (at most `max_hops`).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Hops computed or skipped since the start.
    pub fn total_hops(&self) -> u64 {
        self.total
    }

    /// The kept raw hops `[len][43]`, oldest first.
    pub fn hops(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len * TURN_CHANNELS);
        for i in 0..self.len {
            let slot = (self.head + i) % self.max_hops;
            out.extend_from_slice(&self.ring[slot * TURN_CHANNELS..(slot + 1) * TURN_CHANNELS]);
        }
        out
    }
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
            re[lm.fft.slot(i)] = x[i];
        }
        lm.fft.run(&mut re, &mut im);
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

    /// A glottal-like pulse train: a 1.5 ms Hann pulse every `1 / f0` s (fractional periods).
    fn pulses(f0: f64, secs: f64) -> Vec<f32> {
        let n = (SAMPLE_RATE as f64 * secs) as usize;
        let period = SAMPLE_RATE as f64 / f0;
        let width = 24.0;
        (0..n)
            .map(|i| {
                let t = (i as f64).rem_euclid(period);
                if t < width {
                    (0.4 * (0.5 - 0.5 * (std::f64::consts::TAU * t / width).cos())) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// xorshift white noise in `[-0.5, 0.5)`.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// docs/s6-ear2-prereg.md §4.5: the f0 of a pulse train at 100, 200 and 300 Hz within 1%.
    #[test]
    fn pitch_of_pulse_trains() {
        let mut fe = TurnFrontEnd::new();
        for f0 in [100.0f64, 200.0, 300.0] {
            let (h, n) = fe.hops(&pulses(f0, 1.0), 0);
            assert!(n > 50);
            for row in h.as_chunks::<TURN_CHANNELS>().0.iter() {
                let got = 100.0 * 2f64.powf(row[CH_F0] as f64);
                assert!(
                    (got / f0 - 1.0).abs() < 0.01,
                    "{f0} Hz: {got} Hz, voicing {}",
                    row[CH_VOICING]
                );
                assert!(
                    row[CH_VOICING] > 0.9,
                    "{f0} Hz: voicing {}",
                    row[CH_VOICING]
                );
            }
        }
    }

    /// docs/s6-ear2-prereg.md §4.5: the voicing of white noise below 0.3; silence is unvoiced.
    #[test]
    fn noise_and_silence_are_unvoiced() {
        let mut fe = TurnFrontEnd::new();
        let (h, n) = fe.hops(&noise(2 * SAMPLE_RATE, 9), 0);
        let v: Vec<f32> = h
            .as_chunks::<TURN_CHANNELS>()
            .0
            .iter()
            .map(|r| r[CH_VOICING])
            .collect();
        let worst = v.iter().copied().fold(0.0f32, f32::max);
        let mean = v.iter().sum::<f32>() / n as f32;
        println!("white noise voicing: mean {mean:.3}, max {worst:.3} over {n} hops");
        assert!(worst < 0.3, "max voicing {worst}");
        let (h, _) = fe.hops(&vec![0.0; 3200], 0);
        for r in h.as_chunks::<TURN_CHANNELS>().0.iter() {
            assert_eq!((r[CH_F0], r[CH_VOICING]), (0.0, 0.0));
            assert!(r[..N_MELS].iter().all(|&x| (x + 8.0).abs() < 1e-6));
            assert!((r[CH_ENERGY] + 8.0).abs() < 1e-6);
        }
    }

    /// docs/s6-ear2-prereg.md §4.5: streaming and batch features agree bit for bit, whatever
    /// the chunk sizes, and the stream keeps the last hops only.
    #[test]
    fn stream_matches_batch_bit_for_bit() {
        let mut pcm = pulses(170.0, 2.3);
        for (x, y) in pcm.iter_mut().zip(noise(40_000, 3)) {
            *x += 0.05 * y;
        }
        let mut fe = TurnFrontEnd::new();
        let (all, n) = fe.hops(&pcm, 0);
        assert_eq!(n, turn_hop_count(pcm.len()));
        let (tail, k) = fe.hops(&pcm, 100);
        assert_eq!((k, &tail[..]), (100, &all[(n - 100) * TURN_CHANNELS..]));
        for (keep, chunks) in [
            (
                TURN_WINDOW_HOPS,
                vec![1usize, 7, 159, 160, 161, 640, 1000, 3],
            ),
            (50, vec![16_000, 333, 5]),
            (TURN_WINDOW_HOPS, vec![40_000]),
        ] {
            let mut s = FeatureStream::new(keep);
            let mut at = 0;
            let mut c = 0;
            while at < pcm.len() {
                let len = chunks[c % chunks.len()].min(pcm.len() - at);
                s.push(&pcm[at..at + len]);
                at += len;
                c += 1;
            }
            let want = n.min(keep);
            assert_eq!((s.len(), s.total_hops()), (want, n as u64));
            let got = s.hops();
            let exp = &all[(n - want) * TURN_CHANNELS..];
            assert!(
                got.iter().zip(exp).all(|(a, b)| a.to_bits() == b.to_bits()),
                "stream (keep {keep}) differs from batch"
            );
        }
        let mut s = FeatureStream::new(10);
        s.push(&pcm[..639]);
        assert!(s.is_empty());
        s.push(&pcm[639..640]);
        assert_eq!(s.len(), 1);
        s.reset();
        assert_eq!((s.len(), s.total_hops()), (0, 0));
    }

    #[test]
    fn window_normalisation_follows_whispers_rule() {
        let mut fe = TurnFrontEnd::new();
        let pcm = pulses(220.0, 1.0);
        let (h, n) = fe.hops(&pcm, 0);
        // 97 hops, 4 per position: 24 positions on the last 96 hops
        let (x, p) = normalize_window(&h, n, 4);
        assert_eq!((n, p, x.len()), (97, 24, 24 * 4 * TURN_CHANNELS));
        let raw = &h[TURN_CHANNELS..];
        let m = raw
            .as_chunks::<TURN_CHANNELS>()
            .0
            .iter()
            .flat_map(|r| r[..N_MELS].iter().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        let mut top = f32::NEG_INFINITY;
        for (a, b) in x
            .as_chunks::<TURN_CHANNELS>()
            .0
            .iter()
            .zip(raw.as_chunks::<TURN_CHANNELS>().0.iter())
        {
            for j in 0..N_MELS {
                assert!(a[j] >= -1.0 - 1e-6 && a[j] <= 1.0 + 1e-6);
                assert_eq!(a[j], (b[j].max(m - 8.0) - m + 4.0) / 4.0);
                top = top.max(a[j]);
            }
            assert_eq!(a[CH_ENERGY], (b[CH_ENERGY] - m + 4.0) / 4.0);
            assert_eq!((a[CH_F0], a[CH_VOICING]), (b[CH_F0], b[CH_VOICING]));
        }
        assert_eq!(top, 1.0);
        // a gain does not move the normalized mel values (up to the f32 rounding of the log)
        let loud: Vec<f32> = pcm.iter().map(|v| v * 2.0).collect();
        let (h2, _) = fe.hops(&loud, 0);
        let (x2, _) = normalize_window(&h2, n, 4);
        for (a, b) in x
            .as_chunks::<TURN_CHANNELS>()
            .0
            .iter()
            .zip(x2.as_chunks::<TURN_CHANNELS>().0.iter())
        {
            assert!(a[..N_MELS]
                .iter()
                .zip(&b[..N_MELS])
                .all(|(u, v)| (u - v).abs() < 1e-5));
        }
        assert_eq!(normalize_window(&h[..3 * TURN_CHANNELS], 3, 4).1, 0);
    }

    /// The v1 log-mel path through the shared FFT is unchanged (its parity checksum is in
    /// crates/sky-web/tests/parity/expected.json); the v2 mel channels are v1's before its map.
    #[test]
    fn turn_mel_channels_are_v1_values_before_the_map() {
        let pcm = sine(523.25, 0.5);
        let mel = LogMel::new().log_mel(&pcm);
        let mut fe = TurnFrontEnd::new();
        let (h, n) = fe.hops(&pcm, 0);
        // hop t of v2 reads the 25 ms window that starts at t·160 + 120 samples
        for t in [0usize, 10, n - 1] {
            let start = t * HOP + MEL_OFFSET;
            let v1 = LogMel::new().log_mel(&pcm[start..start + WIN]);
            for j in 0..N_MELS {
                let want = v1[j] * 4.0 - 2.0;
                assert!((h[t * TURN_CHANNELS + j] - want).abs() < 1e-5);
            }
        }
        assert_eq!(mel.len(), 50 * N_MELS);
    }
}
