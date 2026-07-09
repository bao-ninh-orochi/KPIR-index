//! Client-decode microbenchmark: the online recover cost — subtract the
//! precomputed `h_s`, round each of the `R` response cells to a plaintext
//! byte, and scan the `≤ 2ε+3` candidate entries for the fingerprint
//! match. Reports decodes/second.
//!
//! Run: `cargo bench -p kpir-index --bench kpir_decode -- --m 1000000 --value-bytes 32`

#[path = "helpers.rs"]
mod helpers;

use std::io::Write;

fn main() {
    let cli = if helpers::is_bench_invocation() {
        helpers::parse_cli()
    } else {
        helpers::smoke_cli()
    };

    let (server, client, queries) = helpers::setup(&cli);
    let shape = *server.shape();
    helpers::verify(&server, &client, cli.m, cli.value_bytes);

    // Pre-compute answers for the query batch, so decode timing excludes
    // the server answer.
    let answers: Vec<Vec<u32>> = queries.iter().map(|q| server.answer(q)).collect();
    let keys: Vec<[u8; 8]> = (0..queries.len() as u64).map(|i| i.to_le_bytes()).collect();

    let mut idx = 0usize;
    let stats = helpers::measure(
        || {
            let v = client.recover(&keys[idx], &answers[idx]);
            idx = (idx + 1) % answers.len();
            helpers::keep_black_box_used(v);
        },
        cli.samples,
    );

    println!(
        "kpir_decode  m={} l={}B N={}  Dec={:.4} ms ({:.1} dec/s)  R={} cells",
        cli.m,
        cli.value_bytes,
        cli.lwe_dim,
        stats.mean_ms(),
        stats.mean_ops,
        shape.response_dim(),
    );

    let mut csv = helpers::csv_writer(
        "kpir_decode",
        "scheme,m,value_bytes,epsilon,lwe_dim,response_cells,\
         mean_decode_ms,mean_dps,min_dps,max_dps,stddev_dps",
    );
    writeln!(
        csv,
        "kpir_index,{},{},{},{},{},{:.6},{:.2},{:.2},{:.2},{:.2}",
        cli.m,
        cli.value_bytes,
        cli.epsilon,
        cli.lwe_dim,
        shape.response_dim(),
        stats.mean_ms(),
        stats.mean_ops,
        stats.min_ops,
        stats.max_ops,
        stats.stddev_ops,
    )
    .expect("write csv row");
}
