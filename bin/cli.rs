use clap::Parser;
use nihility_gsv::{NihilityGsvConfig, NihilityGsvInferParam, tch};
use std::io;
use std::io::Write;

#[derive(Parser, Debug, Clone)]
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
    #[arg(long, default_value = "15")]
    top_k: i64,
    #[arg(short, long, default_value = "default")]
    selected_model: String,
    #[arg(short, long, default_value = "output")]
    output_dir: String,
    #[arg(short, long)]
    text: Option<String>,
}

fn main() {
    nihility_log::init().expect("could not init log");
    let mut args = CliArgs::parse();
    args.text = Some(
        args.text
            .clone()
            .unwrap_or_else(|| {
                print!("Enter text: ");
                io::stdout().flush().expect("failed to flush stdout");
                let mut input = String::new();
                io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
                input
            })
            .trim()
            .to_string(),
    );
    if args.text.clone().unwrap().is_empty() {
        eprintln!("text: empty");
        return;
    }

    let device = tch::Device::cuda_if_available();
    let gsv = NihilityGsvConfig::from(args.clone())
        .init(device)
        .expect("Failed to init gsv");
    gsv.infer_out_to_local(NihilityGsvInferParam::from(args))
        .expect("Failed to infer gsv");
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

impl From<CliArgs> for NihilityGsvInferParam {
    fn from(value: CliArgs) -> Self {
        NihilityGsvInferParam {
            text: value.text.expect("infer text not init"),
            top_k: value.top_k,
        }
    }
}
