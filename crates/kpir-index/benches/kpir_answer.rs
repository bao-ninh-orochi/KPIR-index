//! Server-answer microbenchmark: the online `ans = qu·D` cost, the
//! dominant term of KPIR^index's per-query latency. One config = one CSV
//! row (throughput in answers/second plus the query/response wire sizes).
//!
//! Run: `cargo bench -p kpir-index --bench kpir_answer -- --m 1000000 --value-bytes 256`

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

    let mut idx = 0usize;
    let stats = helpers::measure("kpir_answer", &cli, || {
        let ans = server.answer(&queries[idx]);
        idx = (idx + 1) % queries.len();
        helpers::keep_black_box_used(ans);
    });

    println!(
        "kpir_answer  m={} l={}B N={} pt={}  Ans={:.2} ms ({:.1} ans/s)  Qry={} Rsp={}",
        cli.m,
        cli.value_bytes,
        cli.lwe_dim,
        shape.plaintext_bits,
        stats.mean_ms(),
        stats.mean_ops,
        helpers::fmt_bytes(shape.query_bytes()),
        helpers::fmt_bytes(shape.response_bytes()),
    );

    let mut csv = helpers::csv_writer(
        "kpir_answer",
        "scheme,m,value_bytes,epsilon,lwe_dim,plaintext_bits,columns,rows,partition,\
         query_bytes,response_bytes,mean_ans_ms,mean_qps,min_qps,max_qps,stddev_qps",
    );
    writeln!(
        csv,
        "kpir_index,{},{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.2},{:.2},{:.2}",
        cli.m,
        cli.value_bytes,
        cli.epsilon,
        cli.lwe_dim,
        shape.plaintext_bits,
        shape.columns,
        shape.rows,
        shape.partition,
        shape.query_bytes(),
        shape.response_bytes(),
        stats.mean_ms(),
        stats.mean_ops,
        stats.min_ops,
        stats.max_ops,
        stats.stddev_ops,
    )
    .expect("write csv row");
}
