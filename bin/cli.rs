use std::io;
use std::io::Write;
use clap::Parser;
use nihility_gsv::{NihilityGsvConfig, tch};
use time::format_description::well_known::Iso8601;
use tracing::metadata::LevelFilter;
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, fmt};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CliArgs {
    #[arg(long, default_value = "base/g2p-en.pt")]
    g2p_en_model: String,
    #[arg(long, default_value = "base/g2p-zh.pt")]
    g2p_zh_model: String,
    #[arg(long, default_value = "base/bert.pt")]
    bert_model: String,
    #[arg(long, default_value = "base/ssl.pt")]
    ssl_model: String,
    #[arg(long, default_value = "model")]
    gsv_dir: String,
    #[arg(short, long, default_value = "default")]
    selected_model: String,
    #[arg(short, long, default_value = "output")]
    output_dir: String,
    #[arg(short, long)]
    text: Option<String>,
}

fn main() {
    let args = CliArgs::parse();
    let text = args.text.clone().unwrap_or_else(|| {
        print!("Enter text: ");
        io::stdout().flush().expect("failed to flush stdout");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        input
    }).trim().to_string();
    if text.is_empty() {
        eprintln!("text: empty");
        return;
    }

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

    let device = tch::Device::cuda_if_available();
    let gsv = NihilityGsvConfig::from(args)
        .init(device)
        .expect("Failed to init gsv");
    gsv.infer_out_to_local(&text).expect("Failed to infer gsv");
}

impl From<CliArgs> for NihilityGsvConfig {
    fn from(value: CliArgs) -> Self {
        NihilityGsvConfig {
            g2p_en_model: value.g2p_en_model,
            g2p_zh_model: value.g2p_zh_model,
            bert_model: value.bert_model,
            ssl_model: value.ssl_model,
            gsv_dir: value.gsv_dir,
            selected_model: value.selected_model,
            output_dir: value.output_dir,
        }
    }
}
