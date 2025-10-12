use chrono::Local;
use clap::Parser;
use nihility_gsv::{NihilityGsvConfig, NihilityGsvInferParam, tch};
use std::io::Write;
use std::path::Path;
use std::{fs, io};
use tracing::{error, info};

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct CliArgs {
    #[arg(long, default_value = "15")]
    top_k: i64,
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

    let output_path = Path::new(&args.output_dir);
    if !output_path.exists() {
        info!("Creating GSV output dir {:?}", output_path);
        fs::create_dir_all(output_path).expect("Could not create output directory");
    } else if !output_path.is_dir() {
        error!("Output path is not a directory");
        return;
    }

    let device = tch::Device::cuda_if_available();
    let gsv = nihility_config::get_config::<NihilityGsvConfig>(env!("CARGO_PKG_NAME").to_string())
        .expect("could not get inner config")
        .init(device)
        .expect("Failed to init gsv");
    let out_wav_name = Local::now().format("%Y-%m-%d-%H-%M-%S").to_string();
    let output = output_path.join(format!("{}.wav", out_wav_name));
    let mut file_out = fs::File::create(&output).expect("Could not create output file");
    let audio_wav_data = gsv
        .infer_out_to_wav(NihilityGsvInferParam::from(args))
        .expect("Failed to infer gsv");
    file_out
        .write_all(&audio_wav_data)
        .expect("Failed to write audio output to file");
    info!("write output audio wav to {}", output.display());
}

impl From<CliArgs> for NihilityGsvInferParam {
    fn from(value: CliArgs) -> Self {
        NihilityGsvInferParam {
            text: value.text.expect("infer text not init"),
            top_k: value.top_k,
        }
    }
}
