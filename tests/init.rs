use std::fs::File;
use std::io::Read;
use sha2::{Digest, Sha256};
use time::format_description::well_known::Iso8601;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{fmt, Layer};
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use nihility_gsv::init::init_libtorch;

#[test]
pub fn get_libtorch_zip_file_sha256() {
    let mut libtorch_zip = File::open("../lib/.tmp/libtorch-cpu.zip").expect("open libtorch_zip failed");
    let mut libtorch_zip_data = Vec::new();
    libtorch_zip.read_to_end(&mut libtorch_zip_data).expect("read libtorch_zip failed");
    let sha256 = hex::encode(Sha256::digest(libtorch_zip_data));
    println!("libtorch_zip sha256: {}", sha256);
}

#[test]
pub fn test_init_libtorch() {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_ansi(false)
                .with_thread_ids(true)
                .with_target(true)
                .with_timer(LocalTime::new(Iso8601::DATE_TIME_OFFSET))
                .with_filter(LevelFilter::DEBUG),
        )
        .init();
    init_libtorch().expect("init_libtorch");
}