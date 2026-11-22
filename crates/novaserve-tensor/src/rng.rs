//! Deterministic, zero-dependency PRNGs (`SplitMix64` and `ChaCha8Rng`) for
//! reproducible weight synthesis, sampling, and adversarial fuzzing.

/// Fast 64-bit SplitMix64 generator used for key scheduling and test fuzzing.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform float in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32_unit(&mut self) -> f32 {
        let bits = (self.next_u32() >> 8) as f32;
        bits * (1.0 / 16_777_216.0)
    }

    /// Uniform double in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64_unit(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) as f64;
        bits * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform float in `[low, high)`.
    #[inline]
    pub fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32_unit()
    }

    /// Standard normal variate via Box-Muller transform.
    pub fn next_gaussian_f32(&mut self) -> f32 {
        let u1 = (self.next_f64_unit()).max(1e-15);
        let u2 = self.next_f64_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos()) as f32
    }
}

/// Cryptographically strong, deterministic 8-round ChaCha PRNG (`ChaCha8Rng`)
/// implementing RFC 7539 quarter-rounds for statistically unbiased sampling.
#[derive(Clone, Debug)]
pub struct ChaCha8Rng {
    key: [u32; 8],
    counter: u64,
    stream: u64,
    buffer: [u32; 16],
    index: usize,
}

impl ChaCha8Rng {
    pub fn from_seed_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut key = [0u32; 8];
        for word in &mut key {
            *word = sm.next_u32();
        }
        let stream = sm.next_u64();
        let mut rng = Self {
            key,
            counter: 0,
            stream,
            buffer: [0u32; 16],
            index: 16,
        };
        rng.refill();
        rng
    }

    pub fn from_key_and_stream(key: [u32; 8], stream: u64) -> Self {
        let mut rng = Self {
            key,
            counter: 0,
            stream,
            buffer: [0u32; 16],
            index: 16,
        };
        rng.refill();
        rng
    }

    #[inline]
    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(16);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(12);

        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(8);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(7);
    }

    fn refill(&mut self) {
        let initial: [u32; 16] = [
            0x6170_7865,
            0x3320_646E,
            0x7962_2D32,
            0x6B20_6574,
            self.key[0],
            self.key[1],
            self.key[2],
            self.key[3],
            self.key[4],
            self.key[5],
            self.key[6],
            self.key[7],
            self.counter as u32,
            (self.counter >> 32) as u32,
            self.stream as u32,
            (self.stream >> 32) as u32,
        ];
        let mut working = initial;
        for _ in 0..4 {
            // Column rounds
            Self::quarter_round(&mut working, 0, 4, 8, 12);
            Self::quarter_round(&mut working, 1, 5, 9, 13);
            Self::quarter_round(&mut working, 2, 6, 10, 14);
            Self::quarter_round(&mut working, 3, 7, 11, 15);
            // Diagonal rounds
            Self::quarter_round(&mut working, 0, 5, 10, 15);
            Self::quarter_round(&mut working, 1, 6, 11, 12);
            Self::quarter_round(&mut working, 2, 7, 8, 13);
            Self::quarter_round(&mut working, 3, 4, 9, 14);
        }
        for i in 0..16 {
            self.buffer[i] = working[i].wrapping_add(initial[i]);
        }
        self.counter = self.counter.wrapping_add(1);
        self.index = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 16 {
            self.refill();
        }
        let out = self.buffer[self.index];
        self.index += 1;
        out
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Uniform float in `[0.0, 1.0)` with 24-bit mantissa precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u32() >> 8) as f32;
        bits * (1.0 / 16_777_216.0)
    }

    /// Uniform double in `[0.0, 1.0)` with 53-bit mantissa precision.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) as f64;
        bits * (1.0 / 9_007_199_254_740_992.0)
    }
}
