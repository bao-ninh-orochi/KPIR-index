//! Approximate key-to-index mapping via optimal Piece-wise Linear
//! Approximation (PLA / PGM).
//!
//! # Purpose
//!
//! `AKIM` (Def. 5 of Hao et al.): given a sorted set of `n` keys, learn a
//! compact structure that, for any key `k`, returns a position within `ε`
//! of `k`'s true rank. The keyword-PIR layer uses it to pick which
//! database column to privately retrieve.
//!
//! # Algorithm
//!
//! The optimal ε-PLA of Ferragina & Vinciguerra (the PGM-index), a direct
//! port of the reference `mpc4j` `PlaModel`. Each key contributes an
//! upper point `(k, i+ε)` and a lower point `(k, i−ε)`; the greedy
//! convex-hull learner extends a segment while a single line can pass
//! within `ε` of every point (the feasible slope cone stays non-empty),
//! and starts a new segment otherwise. It emits the minimum number of
//! segments `d ≤ n/2ε` (Lemma 1); on hashed (near-uniform) keys `d ≈
//! 0.013n` in practice.
//!
//! A segment is a triple `(first_key, slope, intercept)` defining
//! `f(k) = slope·(k − first_key) + intercept`. **Extract** binary-searches
//! for the segment covering `k` and evaluates `f` (matching `mpc4j`'s
//! single-level lookup; the Java build additionally stacks recursive PGM
//! levels purely to speed the segment search — an optimisation orthogonal
//! to the ε guarantee and the head-to-head metrics).
//!
//! # Design notes
//!
//! Geometry uses `f64` for the key coordinate and `i64` for the index
//! coordinate, exactly as `mpc4j` (`Point{double x; long y}`). For 64-bit
//! hashed keys the `u64 → f64` mantissa truncation is immaterial: with
//! `n ≪ 2^53` uniformly spread keys, distinct keys never collide in `f64`.

/// One PLA segment: `f(k) = slope·(k − first_key) + intercept`.
#[derive(Clone, Copy, Debug)]
struct Segment {
    first_key: u64,
    slope: f64,
    intercept: i64,
}

/// A learned ε-approximate key-to-index mapping over `n` sorted keys.
#[derive(Clone, Debug)]
pub struct KeyIndexMap {
    segments: Vec<Segment>,
    n: usize,
    epsilon: u32,
}

impl KeyIndexMap {
    /// On-wire / in-memory footprint of one segment: `first_key` (8) +
    /// `slope` (8) + `intercept` (8) = 24 bytes (matches `mpc4j`).
    pub const SEGMENT_BYTES: usize = 24;

    /// Learn the map over `sorted_keys` (strictly increasing, `u64`).
    ///
    /// # Constraints
    ///
    /// Panics if `sorted_keys` is not strictly increasing.
    pub fn build(sorted_keys: &[u64], epsilon: u32) -> Self {
        let n = sorted_keys.len();
        let mut builder = PlaBuilder::new(epsilon);
        let mut segments = Vec::new();
        for (i, &key) in sorted_keys.iter().enumerate() {
            if i > 0 {
                assert!(
                    key > sorted_keys[i - 1],
                    "keys must be strictly increasing (dup/unsorted at index {i})"
                );
            }
            builder.add_key(key, i as i64, &mut segments);
        }
        if n > 0 {
            builder.finish(&mut segments);
        }
        Self {
            segments,
            n,
            epsilon,
        }
    }

    /// Extract the approximate position of `key`: a point estimate in
    /// `[0, n)` guaranteed within `ε` of the true rank for any key that
    /// was in the training set. For an out-of-range key the result is a
    /// best-effort clamp (the keyword layer's fingerprint check rejects a
    /// wrong hit).
    pub fn extract(&self, key: u64) -> usize {
        if self.n == 0 {
            return 0;
        }
        // Last segment whose first_key ≤ key (segments are in increasing
        // first_key order). `partition_point` returns the count of leading
        // segments satisfying the predicate.
        let idx = self
            .segments
            .partition_point(|s| s.first_key <= key)
            .saturating_sub(1);
        let seg = &self.segments[idx];
        let delta = (key as f64) - (seg.first_key as f64);
        let pos = seg.slope * delta + seg.intercept as f64;
        // Truncate toward zero (mpc4j `(int)` cast), then clamp to [0, n-1].
        let pos = pos as i64;
        pos.clamp(0, self.n as i64 - 1) as usize
    }

    /// Number of learned segments `d`.
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Number of keys `n` the map was built over.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the map is empty (built over zero keys).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The error bound `ε`.
    #[inline]
    pub fn epsilon(&self) -> u32 {
        self.epsilon
    }

    /// Client-side map size in bytes (`d · 24`), the KPIR^index client's
    /// key-to-index memory.
    #[inline]
    pub fn wire_byte_size(&self) -> usize {
        self.segments.len() * Self::SEGMENT_BYTES
    }
}

/// A `(dx, dy)` slope, `dx` continuous, `dy` integral — mirrors the
/// `mpc4j` `Slope{double dx; long dy}`, so comparisons use the same
/// mixed-precision cross products.
#[derive(Clone, Copy)]
struct Slope {
    dx: f64,
    dy: i64,
}

impl Slope {
    #[inline]
    fn between(p1: Point, p2: Point) -> Slope {
        Slope {
            dx: p2.x - p1.x,
            dy: p2.y - p1.y,
        }
    }
    /// `self < other` ⟺ `dy·other.dx < dx·other.dy`.
    #[inline]
    fn is_less_than(self, o: Slope) -> bool {
        (self.dy as f64) * o.dx < self.dx * (o.dy as f64)
    }
    #[inline]
    fn is_greater_than(self, o: Slope) -> bool {
        (self.dy as f64) * o.dx > self.dx * (o.dy as f64)
    }
    #[inline]
    fn is_equal(self, o: Slope) -> bool {
        (self.dy as f64) * o.dx == self.dx * (o.dy as f64)
    }
    #[inline]
    fn as_double(p1: Point, p2: Point) -> f64 {
        (p2.y - p1.y) as f64 / (p2.x - p1.x)
    }
}

/// A `(x, y)` point: `x` the key coordinate (`f64`), `y` the (± ε) index.
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: i64,
}

#[inline]
fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.x - o.x) * ((b.y - o.y) as f64) - ((a.y - o.y) as f64) * (b.x - o.x)
}

/// Streaming optimal ε-PLA learner (port of `mpc4j` `PlaModel`).
struct PlaBuilder {
    epsilon: i64,
    first_key: u64,
    num_points_in_hull: usize,
    rect: [Point; 4],
    lower: Vec<Point>,
    upper: Vec<Point>,
    lower_start: usize,
    upper_start: usize,
}

impl PlaBuilder {
    fn new(epsilon: u32) -> Self {
        let zero = Point { x: 0.0, y: 0 };
        Self {
            epsilon: epsilon as i64,
            first_key: 0,
            num_points_in_hull: 0,
            rect: [zero; 4],
            lower: Vec::with_capacity(256),
            upper: Vec::with_capacity(256),
            lower_start: 0,
            upper_start: 0,
        }
    }

    fn add_key(&mut self, key: u64, index: i64, out: &mut Vec<Segment>) {
        let x = key as f64;
        let p1 = Point {
            x,
            y: index + self.epsilon,
        }; // upper
        let p2 = Point {
            x,
            y: index - self.epsilon,
        }; // lower

        if self.num_points_in_hull > 1 {
            let slope1 = Slope::between(self.rect[0], self.rect[2]);
            let slope2 = Slope::between(self.rect[1], self.rect[3]);
            let outside1 = Slope::between(self.rect[2], p1).is_less_than(slope1);
            let outside2 = Slope::between(self.rect[3], p2).is_greater_than(slope2);
            if outside1 || outside2 {
                self.produce_segment(out);
                self.num_points_in_hull = 0;
            }
        }

        if self.num_points_in_hull == 0 {
            self.first_key = key;
            self.rect[0] = p1;
            self.rect[1] = p2;
            self.upper.clear();
            self.lower.clear();
            self.upper.push(p1);
            self.lower.push(p2);
            self.upper_start = 0;
            self.lower_start = 0;
            self.num_points_in_hull = 1;
            return;
        }
        if self.num_points_in_hull == 1 {
            self.rect[2] = p2;
            self.rect[3] = p1;
            self.upper.push(p1);
            self.lower.push(p2);
            self.num_points_in_hull = 2;
            return;
        }

        // num_points_in_hull ≥ 2 and point is inside the cone: extend.
        let slope1 = Slope::between(self.rect[0], self.rect[2]);
        let slope2 = Slope::between(self.rect[1], self.rect[3]);

        if Slope::between(self.rect[1], p1).is_less_than(slope2) {
            // Tighten the upper boundary: new rect[1], rect[3].
            let mut slope_min = Slope::between(p1, self.lower[self.lower_start]);
            let mut min_i = self.lower_start;
            for i in (self.lower_start + 1)..self.lower.len() {
                let s = Slope::between(p1, self.lower[i]);
                if s.is_greater_than(slope_min) {
                    break;
                }
                slope_min = s;
                min_i = i;
            }
            self.rect[1] = self.lower[min_i];
            self.rect[3] = p1;
            self.lower_start = min_i;

            let mut end = self.upper.len();
            while end >= self.upper_start + 2
                && cross(self.upper[end - 2], self.upper[end - 1], p1) <= 0.0
            {
                end -= 1;
            }
            self.upper.truncate(end);
            self.upper.push(p1);
        }

        if Slope::between(self.rect[0], p2).is_greater_than(slope1) {
            // Tighten the lower boundary: new rect[0], rect[2].
            let mut slope_max = Slope::between(p2, self.upper[self.upper_start]);
            let mut max_i = self.upper_start;
            for i in (self.upper_start + 1)..self.upper.len() {
                let s = Slope::between(p2, self.upper[i]);
                if s.is_less_than(slope_max) {
                    break;
                }
                slope_max = s;
                max_i = i;
            }
            self.rect[0] = self.upper[max_i];
            self.rect[2] = p2;
            self.upper_start = max_i;

            let mut end = self.lower.len();
            while end >= self.lower_start + 2
                && cross(self.lower[end - 2], self.lower[end - 1], p2) >= 0.0
            {
                end -= 1;
            }
            self.lower.truncate(end);
            self.lower.push(p2);
        }

        self.num_points_in_hull += 1;
    }

    fn produce_segment(&self, out: &mut Vec<Segment>) {
        let (slope, intercept) = if self.num_points_in_hull == 1 {
            (0.0, (self.rect[0].y + self.rect[1].y) >> 1)
        } else {
            let (p0, p1, p2, p3) = (self.rect[0], self.rect[1], self.rect[2], self.rect[3]);
            let slope1 = Slope::between(p0, p2);
            let slope2 = Slope::between(p1, p3);
            let (intersect_x, intersect_y) = if slope1.is_equal(slope2) {
                (p0.x, p0.y as f64)
            } else {
                let s01 = Slope::between(p0, p1);
                let a = slope1.dx * (slope2.dy as f64) - (slope1.dy as f64) * slope2.dx;
                let b = (s01.dx * (slope2.dy as f64) - (s01.dy as f64) * slope2.dx) / a;
                (p0.x + b * slope1.dx, p0.y as f64 + b * (slope1.dy as f64))
            };
            let min_slope = Slope::as_double(p0, p2);
            let max_slope = Slope::as_double(p1, p3);
            let slope = (min_slope + max_slope) / 2.0;
            let intercept = (intersect_y - (intersect_x - self.first_key as f64) * slope) as i64;
            (slope, intercept)
        };
        out.push(Segment {
            first_key: self.first_key,
            slope,
            intercept,
        });
    }

    fn finish(&mut self, out: &mut Vec<Segment>) {
        self.produce_segment(out);
        self.num_points_in_hull = 0;
        self.lower.clear();
        self.upper.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    /// The Extract guarantee: for every training key, Extract is within
    /// `ε + 1` of its true rank. The extra `+1` is the `mpc4j`-faithful
    /// intercept truncation (`(long)` cast) — which is exactly why the
    /// keyword layer pads each column by `ε+1` rows on top and `ε+2` on
    /// the bottom (`rows = dataRows + 2ε + 3`). Checked across several key
    /// distributions.
    fn assert_eps_guarantee(keys: &[u64], epsilon: u32) {
        let map = KeyIndexMap::build(keys, epsilon);
        let bound = epsilon as i64 + 1;
        for (i, &k) in keys.iter().enumerate() {
            let pos = map.extract(k) as i64;
            let err = (pos - i as i64).abs();
            assert!(
                err <= bound,
                "extract bound violated: key #{i}={k} extract={pos} err={err} > {bound} \
                 (segments={})",
                map.num_segments()
            );
        }
    }

    #[test]
    fn eps_guarantee_sequential() {
        let keys: Vec<u64> = (0..5000).map(|i| i as u64 * 7 + 3).collect();
        assert_eps_guarantee(&keys, 4);
    }

    #[test]
    fn eps_guarantee_uniform_hashed() {
        let mut rng = StdRng::seed_from_u64(0xF00D);
        let mut keys: Vec<u64> = (0..20_000).map(|_| rng.next_u64()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eps_guarantee(&keys, 4);
    }

    #[test]
    fn eps_guarantee_clustered() {
        // Dense clusters separated by large gaps — stresses segment splits.
        let mut keys = Vec::new();
        for cluster in 0..200u64 {
            let base = cluster * 1_000_000;
            for j in 0..100u64 {
                keys.push(base + j);
            }
        }
        keys.sort_unstable();
        keys.dedup();
        assert_eps_guarantee(&keys, 4);
    }

    #[test]
    fn eps_guarantee_various_epsilon() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut keys: Vec<u64> = (0..10_000).map(|_| rng.next_u64() >> 8).collect();
        keys.sort_unstable();
        keys.dedup();
        for eps in [0u32, 1, 2, 4, 8, 16] {
            assert_eps_guarantee(&keys, eps);
        }
    }

    #[test]
    fn edge_cases() {
        let empty = KeyIndexMap::build(&[], 4);
        assert_eq!(empty.extract(123), 0);
        assert_eq!(empty.num_segments(), 0);

        let one = KeyIndexMap::build(&[42], 4);
        assert_eq!(one.extract(42), 0);
        assert_eq!(one.num_segments(), 1);

        let two = KeyIndexMap::build(&[10, 20], 4);
        assert!(two.extract(10) <= 5 && two.extract(20) <= 5);
    }

    /// Fewer segments than the `n/2ε` bound; near-uniform keys compress well.
    #[test]
    fn segment_count_reasonable() {
        let mut rng = StdRng::seed_from_u64(999);
        let mut keys: Vec<u64> = (0..100_000).map(|_| rng.next_u64()).collect();
        keys.sort_unstable();
        keys.dedup();
        let map = KeyIndexMap::build(&keys, 4);
        assert!(
            map.num_segments() <= keys.len() / 8,
            "too many segments: {}",
            map.num_segments()
        );
    }
}
