//! Client-query microbenchmark: the online cost of turning a keyword into
//! a query — PLA `extract`, column selection, and the LWE encryption of
//! the column selector (`qu = a_s + e + Δ·u_col`). Reports queries/second
//! and the query upload size.
//!
//! Run: `cargo bench -p kpir-index --bench kpir_query -- --m 1000000 --value-bytes 32`

#[path = "helpers.rs"]
mod helpers;

use std::io::Write;

use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let cli = if helpers::is_bench_invocation() {
        helpers::parse_cli()
    } else {
        helpers::smoke_cli()
    };

    let (server, client, _queries) = helpers::setup(&cli);
    let shape = *server.shape();
    helpers::verify(&server, &client, cli.m, cli.value_bytes);

    let mut rng = StdRng::seed_from_u64(cli.seed ^ 0xA5A5);
    let mut i = 0u64;
    let m = cli.m as u64;
    let stats = helpers::measure(
        || {
            let qu = client.query(&i.to_le_bytes(), &mut rng);
            i = (i + 1) % m;
            helpers::keep_black_box_used(qu);
        },
        cli.samples,
    );

    println!(
        "kpir_query  m={} l={}B N={}  Qry={:.4} ms ({:.1} qry/s)  upload={}",
        cli.m,
        cli.value_bytes,
        cli.lwe_dim,
        stats.mean_ms(),
        stats.mean_ops,
        helpers::fmt_bytes(shape.query_bytes()),
    );

    let mut csv = helpers::csv_writer(
        "kpir_query",
        "scheme,m,value_bytes,epsilon,lwe_dim,columns,query_bytes,\
         mean_query_ms,mean_qps,min_qps,max_qps,stddev_qps",
    );
    writeln!(
        csv,
        "kpir_index,{},{},{},{},{},{},{:.6},{:.2},{:.2},{:.2},{:.2}",
        cli.m,
        cli.value_bytes,
        cli.epsilon,
        cli.lwe_dim,
        shape.columns,
        shape.query_bytes(),
        stats.mean_ms(),
        stats.mean_ops,
        stats.min_ops,
        stats.max_ops,
        stats.stddev_ops,
    )
    .expect("write csv row");
}
