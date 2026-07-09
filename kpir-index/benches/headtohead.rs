//! Head-to-head online-cost benchmark — the KPIR^index row of CANS2026
//! Table 3. For one `(m, value_bytes)` config it reports the query upload,
//! the response download, and the mean server answer latency, alongside
//! the setup hint size, the client key-to-index map size, and the
//! database expansion rate. One config = one appended CSV row.
//!
//! Run: `cargo bench -p kpir-index --bench headtohead -- --m 1000000 --value-bytes 32`

#[path = "helpers.rs"]
mod helpers;

use std::io::Write;

fn main() {
    let cli = if helpers::is_bench_invocation() {
        helpers::parse_cli()
    } else {
        // Under `cargo test --all-targets`: smoke, not the full sweep.
        helpers::smoke_cli()
    };
    run(&cli);
}

fn run(cli: &helpers::Cli) {
    let (server, client, queries) = helpers::setup(cli);
    let shape = *server.shape();

    // Correctness gate: never report numbers for a broken pipeline.
    helpers::verify(&server, &client, cli.m, cli.value_bytes);

    // Measure server answer latency, cycling the pre-built queries.
    let mut idx = 0usize;
    let stats = helpers::measure(
        || {
            let ans = server.answer(&queries[idx]);
            idx = (idx + 1) % queries.len();
            helpers::keep_black_box_used(ans);
        },
        cli.samples,
    );

    let query_bytes = shape.query_bytes();
    let response_bytes = shape.response_bytes();
    let hint_bytes = client.hint_wire_bytes();
    let map_bytes = client.map_wire_bytes();

    println!(
        "KPIR^index  m={}  l={}B  N={}  eps={}  pt={}\n  \
         matrix: {}x{} (data_rows={}, rows={}, partition={})  expansion={:.3}\n  \
         Qry={}  Rsp={}  Ans={:.2} ms ({:.1} ans/s)\n  \
         setup hint={}  client map={} ({} segments)",
        cli.m,
        cli.value_bytes,
        cli.lwe_dim,
        cli.epsilon,
        shape.plaintext_bits,
        shape.columns,
        shape.response_dim(),
        shape.data_rows,
        shape.rows,
        shape.partition,
        shape.expansion_rate(),
        helpers::fmt_bytes(query_bytes),
        helpers::fmt_bytes(response_bytes),
        stats.mean_ms(),
        stats.mean_ops,
        helpers::fmt_bytes(hint_bytes),
        helpers::fmt_bytes(map_bytes),
        client.num_segments(),
    );

    let mut csv = helpers::csv_writer(
        "kpir_headtohead",
        "scheme,m,value_bytes,epsilon,lwe_dim,plaintext_bits,columns,data_rows,rows,partition,\
         query_bytes,response_bytes,hint_bytes,map_bytes,num_segments,expansion,\
         mean_ans_ms,answers_per_sec,min_ans_per_sec,max_ans_per_sec,stddev_ans_per_sec",
    );
    writeln!(
        csv,
        "kpir_index,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2}",
        cli.m,
        cli.value_bytes,
        cli.epsilon,
        cli.lwe_dim,
        shape.plaintext_bits,
        shape.columns,
        shape.data_rows,
        shape.rows,
        shape.partition,
        query_bytes,
        response_bytes,
        hint_bytes,
        map_bytes,
        client.num_segments(),
        shape.expansion_rate(),
        stats.mean_ms(),
        stats.mean_ops,
        stats.min_ops,
        stats.max_ops,
        stats.stddev_ops,
    )
    .expect("write csv row");
}
