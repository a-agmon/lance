// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end smoke test for the IVF streaming partition search under a
//! single-thread CPU pool (#7642).
//!
//! The old implementation ran the whole recv/search/send loop inside a single
//! `spawn_cpu` closure with blocking channel operations. If the prefilter's
//! row mask was not yet ready when that closure parked the only pool thread in
//! `blocking_recv`, the mask's own `spawn_cpu` work queued behind it and the
//! producer's `wait_for_ready` never resolved — a permanent 0%-CPU hang.
//!
//! This test runs the full ingredient list on a real index — an HNSW
//! sub-index (no global top-k heap support, so `search_partitions` takes the
//! streaming branch), a prefiltered query on a stable-row-id dataset (the only
//! deletion-mask path that does `spawn_cpu` work on the producer side), and a
//! 1-thread CPU pool. The pool size is a process-global singleton read once
//! from `LANCE_CPU_THREADS`, so the test re-executes itself in a child process
//! with `LANCE_CPU_THREADS=1` and fails if the child does not finish in time.
//!
//! Honest caveat: this does not deterministically reproduce the old deadlock.
//! The hang was a race — the prefilter mask work had to still be pending when
//! the search consumer parked — that real object-storage latency loses but a
//! local filesystem always wins (mask I/O completes in microseconds, and local
//! reads bypass the `ObjectStore` trait, so latency cannot be injected). What
//! this test does guarantee is that the streaming search path, including its
//! channel back-pressure and batch dispatch, completes end to end when the CPU
//! pool has a single thread.

use arrow_array::Float32Array;
use arrow_array::types::{Float32Type, Int32Type};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;

const CHILD_ENV: &str = "LANCE_TEST_IVF_DEADLOCK_CHILD";
const DATASET_ENV: &str = "LANCE_TEST_IVF_DEADLOCK_DATASET";
const DIM: u32 = 16;

/// Runs the prefiltered ANN query that exercises the streaming partition
/// search path.
async fn run_streaming_search(uri: &str) {
    let dataset = Dataset::open(uri).await.unwrap();
    let query = Float32Array::from(vec![0.0f32; DIM as usize]);
    let mut scan = dataset.scan();
    scan.nearest("vec", &query, 10).unwrap();
    scan.minimum_nprobes(1);
    scan.maximum_nprobes(4);
    scan.filter("i < 900").unwrap();
    scan.prefilter(true);
    let batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let num_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(num_rows, 10, "expected k results from the ANN search");
}

#[tokio::test]
async fn test_ivf_streaming_search_completes_on_single_cpu_thread() {
    if std::env::var(CHILD_ENV).is_ok() {
        // Child process: the 1-thread CPU pool is in effect via
        // `LANCE_CPU_THREADS=1`. The search must complete rather than deadlock.
        let uri = std::env::var(DATASET_ENV).expect("child needs the dataset uri");
        run_streaming_search(&uri).await;
        return;
    }

    // Parent process: build the dataset and index with the normal CPU pool so
    // only the search itself runs under the constrained pool.
    let tmp = TempStrDir::default();
    let uri = tmp.as_str().to_owned();
    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
        .into_reader_rows(RowCount::from(1024), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &uri,
        Some(WriteParams {
            // Only the stable-row-id deletion mask builds its allow list via
            // `spawn_cpu`, putting producer-side pool work on the query path.
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // HNSW sub-index: no global top-k heap support, so the initial search
    // takes the streaming partition-search path this test exercises.
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        IvfBuildParams {
            num_partitions: Some(4),
            ..Default::default()
        },
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    // Deleted rows give the prefilter a deletion mask to build.
    dataset.delete("i >= 900").await.unwrap();

    // Sanity check: the query itself is valid under the parent's normal pool.
    run_streaming_search(&uri).await;

    let exe = std::env::current_exe().expect("locate test binary");
    let mut child = std::process::Command::new(exe)
        .args([
            "ivf_stream_deadlock::test_ivf_streaming_search_completes_on_single_cpu_thread",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env(DATASET_ENV, &uri)
        .env("LANCE_CPU_THREADS", "1")
        .spawn()
        .expect("spawn child test process");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        match child.try_wait().expect("poll child process status") {
            Some(status) => {
                assert!(status.success(), "child search process failed: {status}");
                break;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "IVF streaming search stalled under a single-thread CPU pool: \
                     child did not finish within 120s"
                );
            }
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
}
