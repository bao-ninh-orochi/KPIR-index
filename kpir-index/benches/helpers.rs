//! Shared benchmark harness: CLI parsing, a manual throughput sampler,
//! CSV output, and a correctness verifier.
//!
//! Included by each bench via `#[path = "helpers.rs"] mod helpers;`.
//! Mirrors the RisePIR (`ikpir`) bench conventions so head-to-head rows
//! line up: fixed-width wire-byte communication, per-sample timing folded
//! to mean/min/max/stddev ops-per-second, one config = one appended CSV
//! row under `${KPIR_RESULTS_DIR:-results}/<bench>.csv`.
#![allow(dead_code)]

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use criterion::black_box;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kpir_index::scheme::{build_synthetic, hash_key, synthetic_value, KpirClient, KpirServer};
use simplepir::SimpleConfig;

/// Head-to-head benchmark knobs. Value sizes {32, 256, 1024} B and
/// `m = 10^6` reproduce CANS2026 Table 3; `lwe_dim = 1275` matches the
/// SimplePIR backend of RisePIR-S (128-bit security).
#[derive(Parser, Debug, Clone)]
#[command(about = "KPIR^index (Hao et al.) head-to-head benchmark")]
pub struct Cli {
    /// Number of key-value pairs m.
    #[arg(long, default_value_t = 1_000_000)]
    pub m: usize,
    /// Value length ℓ in bytes.
    #[arg(long, default_value_t = 32)]
    pub value_bytes: usize,
    /// Approximation error ε.
    #[arg(long, default_value_t = 4)]
    pub epsilon: u32,
    /// LWE dimension N (1275 = 128-bit, matches RisePIR-S; 1024 = mpc4j default).
    #[arg(long, default_value_t = 1275)]
    pub lwe_dim: u32,
    /// Number of distinct queries cycled through during timing.
    #[arg(long, default_value_t = 16)]
    pub batch: usize,
    /// Number of timing samples per config.
    #[arg(long, default_value_t = 20)]
    pub samples: usize,
    /// RNG seed.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

/// Whether this process was launched by `cargo bench` (which passes
/// `--bench`), as opposed to `cargo test --all-targets`.
pub fn is_bench_invocation() -> bool {
    std::env::args().any(|a| a == "--bench")
}

/// A tiny config used when a bench is run under `cargo test --all-targets`
/// (compile + smoke, not the full sweep).
pub fn smoke_cli() -> Cli {
    Cli {
        m: 2000,
        value_bytes: 32,
        epsilon: 4,
        lwe_dim: 256,
        batch: 4,
        samples: 3,
        seed: 1,
    }
}

/// Parse the CLI, tolerating the `--bench` flag cargo injects.
pub fn parse_cli() -> Cli {
    let args: Vec<String> = std::env::args().filter(|a| a != "--bench").collect();
    Cli::parse_from(args)
}

/// Timing statistics over the collected samples (all in ops/second except
/// `mean_ns`, the mean per-call latency in nanoseconds).
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub mean_ops: f64,
    pub min_ops: f64,
    pub max_ops: f64,
    pub stddev_ops: f64,
    pub mean_ns: f64,
}

impl Stats {
    /// Mean per-call latency in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean_ns / 1.0e6
    }
}

/// Measure `body` (one "operation" per call): warm up, auto-calibrate the
/// inner iteration count to ~50 ms/sample, then collect `samples` samples
/// and reduce to per-second statistics. `black_box` defeats dead-code
/// elimination of the returned value.
pub fn measure<F: FnMut()>(mut body: F, samples: usize) -> Stats {
    // Calibrate on one call.
    let probe = {
        let s = Instant::now();
        body();
        s.elapsed()
    };
    let call_ns = probe.as_nanos().max(1) as u64;
    let iters = (50_000_000u64 / call_ns).max(1);

    // Warm up ~100 ms.
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(100) {
        body();
    }

    let mut ns_per_iter = Vec::with_capacity(samples);
    for _ in 0..samples.max(1) {
        let start = Instant::now();
        for _ in 0..iters {
            body();
        }
        let elapsed = start.elapsed().as_nanos() as f64 / iters as f64;
        ns_per_iter.push(elapsed);
    }

    let ops: Vec<f64> = ns_per_iter.iter().map(|&ns| 1.0e9 / ns).collect();
    let k = ops.len() as f64;
    let mean_ops = ops.iter().sum::<f64>() / k;
    let mean_ns = ns_per_iter.iter().sum::<f64>() / k;
    let min_ops = ops.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ops = ops.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = ops.iter().map(|&o| (o - mean_ops).powi(2)).sum::<f64>() / k;
    Stats {
        mean_ops,
        min_ops,
        max_ops,
        stddev_ops: var.sqrt(),
        mean_ns,
    }
}

/// Open `${KPIR_RESULTS_DIR:-results}/<name>.csv` in append mode, writing
/// `header` first if the file is new/empty. Returns a buffered writer.
pub fn csv_writer(name: &str, header: &str) -> BufWriter<File> {
    let dir = std::env::var("KPIR_RESULTS_DIR").unwrap_or_else(|_| "results".to_string());
    std::fs::create_dir_all(&dir).expect("create results dir");
    let path = PathBuf::from(&dir).join(format!("{name}.csv"));

    let existing_len = File::open(&path)
        .and_then(|mut f| {
            let mut s = String::new();
            f.read_to_string(&mut s).map(|_| s.len())
        })
        .unwrap_or(0);

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("open csv");
    let mut w = BufWriter::new(file);
    if existing_len == 0 {
        writeln!(w, "{header}").expect("write header");
    }
    w
}

/// Build the synthetic KPIR^index database for a config, and pre-build a
/// batch of distinct queries (present keys `0..batch`).
pub fn setup(cli: &Cli) -> (KpirServer, KpirClient, Vec<Vec<u32>>) {
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let config = SimpleConfig::with_lwe_dim(cli.lwe_dim);
    let (server, client) = build_synthetic(
        cli.m,
        cli.value_bytes,
        cli.epsilon,
        &config,
        seed16(cli.seed),
        &mut rng,
    );
    let queries: Vec<Vec<u32>> = (0..cli.batch.max(1))
        .map(|i| client.query(&(i as u64).to_le_bytes(), &mut rng))
        .collect();
    (server, client, queries)
}

/// Verify the pipeline once: a handful of present keys recover their exact
/// synthetic values, and an absent key returns ⊥. Panics on mismatch.
pub fn verify(server: &KpirServer, client: &KpirClient, m: usize, value_bytes: usize) {
    let mut rng = StdRng::seed_from_u64(0xF1ED_BEEF_u64.wrapping_add(m as u64));
    let sample: Vec<u64> = [0u64, 1, (m / 2) as u64, (m - 1) as u64].to_vec();
    for i in sample {
        let key = i.to_le_bytes();
        let qu = client.query(&key, &mut rng);
        let ans = server.answer(&qu);
        let got = client.recover(&key, &ans).expect("present key recovered");
        let mut want = vec![0u8; value_bytes];
        synthetic_value(hash_key(&key), &mut want);
        assert_eq!(got, want, "verification failed for present key {i}");
    }
    // Absent key (m is not one of 0..m as a little-endian u64 label).
    let absent = (m as u64 + 12345).to_le_bytes();
    let qu = client.query(&absent, &mut rng);
    let ans = server.answer(&qu);
    assert!(client.recover(&absent, &ans).is_none(), "absent key not ⊥");
}

/// Human-readable byte size.
pub fn fmt_bytes(bytes: usize) -> String {
    let b = bytes as f64;
    if b >= 1.0e6 {
        format!("{:.2} MB", b / 1.0e6)
    } else if b >= 1.0e3 {
        format!("{:.2} kB", b / 1.0e3)
    } else {
        format!("{bytes} B")
    }
}

/// A 16-byte public seed derived from a u64.
pub fn seed16(seed: u64) -> [u8; 16] {
    let mut s = [0u8; 16];
    s[..8].copy_from_slice(&seed.to_le_bytes());
    s[8..].copy_from_slice(&seed.rotate_left(32).to_le_bytes());
    s
}

/// Keep `black_box` referenced even if a bench doesn't use it directly.
pub fn keep_black_box_used<T>(x: T) -> T {
    black_box(x)
}
