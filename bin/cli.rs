use chrono::Local;
use clap::Parser;
use nihility_gsv::{gsv, tch, text, text::G2PConfig};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use time::format_description::well_known::Iso8601;
use tracing::metadata::LevelFilter;
use tracing::{error, info};
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, fmt};

const G2P_EN_MODEL: &str = "g2p-en.pt";
const G2P_ZH_MODEL: &str = "g2p-zh.pt";
const BERT_MODEL: &str = "bert.pt";
const SSL_MODEL: &str = "ssl.pt";
const T2S_MODEL: &str = "t2s.pt";
const VITS_MODEL: &str = "vits.pt";
const REF_PATH: &str = "ref.wav";
const REF_TEXT: &str = "ref.txt";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CliArgs {
    #[arg(long, default_value = "model/base")]
    base_model_dir: String,
    #[arg(long, default_value = "model")]
    gsv_model_dir: String,
    #[arg(short, long, default_value = "default")]
    model: String,
    #[arg(short, long, default_value = "output")]
    output: String,
    #[arg(short, long)]
    text: String,
}

fn main() {
    let args = CliArgs::parse();
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

    info!("check config: {:?}", &args);
    let base_model_dir = Path::new(&args.base_model_dir);
    let gsv_model_dir = Path::new(&args.gsv_model_dir);
    let gsv_model = gsv_model_dir.join(&args.model);
    if !base_model_dir.exists() {
        error!("Base model dir does not exist");
        return;
    }
    if !gsv_model.exists() {
        error!("GSV model does not exist");
        return;
    }

    let output_path = Path::new(&args.output).join(&args.model);
    if !output_path.exists() {
        info!("Creating output directory");
        fs::create_dir_all(&output_path).expect("Unable to create output directory");
    } else if output_path.is_file() {
        error!("Output path mast be a directory");
        return;
    }

    let g2p_en_path = base_model_dir.join(G2P_EN_MODEL);
    if !g2p_en_path.exists() {
        error!("G2P en model does not exist");
        return;
    }
    let g2p_en_path = g2p_en_path
        .to_str()
        .expect("G2P en path exception")
        .to_string();

    let g2p_zh_path = base_model_dir.join(G2P_ZH_MODEL);
    if !g2p_zh_path.exists() {
        error!("G2P zh model does not exist");
        return;
    }
    let g2p_zh_path = g2p_zh_path
        .to_str()
        .expect("G2P zh path exception")
        .to_string();

    let bert_path = base_model_dir.join(BERT_MODEL);
    if !bert_path.exists() {
        error!("Bert model does not exist");
        return;
    }
    let bert_path = bert_path.to_str().expect("bert path exception").to_string();

    let ssl_path = base_model_dir.join(SSL_MODEL);
    if !ssl_path.exists() {
        error!("SSL model does not exist");
        return;
    }
    let ssl_path = ssl_path.to_str().expect("ssl path exception").to_string();

    let ref_path = gsv_model.join(REF_PATH);
    if !ref_path.exists() {
        error!("Ref audio does not exist");
        return;
    }
    let ref_path = ref_path.to_str().expect("Ref audio exception").to_string();

    let ref_text = gsv_model.join(REF_TEXT);
    if !ref_text.exists() {
        error!("Ref text does not exist");
        return;
    }
    let ref_text = fs::read_to_string(&ref_text).expect("Ref text exception");

    let t2s_path = gsv_model.join(T2S_MODEL);
    if !t2s_path.exists() {
        error!("T2S model does not exist");
        return;
    }
    let t2s_path = t2s_path.to_str().expect("T2S path exception").to_string();

    let vits_path = gsv_model.join(VITS_MODEL);
    if !vits_path.exists() {
        error!("VITS model does not exist");
        return;
    }
    let vits_path = vits_path.to_str().expect("VITS path exception").to_string();

    let g2p_conf = G2PConfig::new(g2p_en_path).with_chinese(g2p_zh_path, bert_path);

    let device = tch::Device::cuda_if_available();
    info!("use torch device: {:?}", device);

    let g2p = g2p_conf.build(device).expect("failed to build g2p");

    let file = fs::File::open(ref_path).unwrap();
    let (head, mut ref_audio_samples) = wav_io::read_from_file(file).unwrap();
    info!("head: {:?}", head);
    if head.sample_rate != 32000 {
        info!("ref audio sample rate: {}, need 32000", head.sample_rate);
        ref_audio_samples = wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32000);
    }
    info!("load ref done");

    let text = args.text.to_string();
    let (text_seq, text_bert) = text::get_phone_and_bert(&g2p, &text).unwrap();

    let (ref_seq, ref_bert) = text::get_phone_and_bert(&g2p, &ref_text).unwrap();

    let ref_audio_32k = tch::Tensor::from_slice(&ref_audio_samples)
        .to_device(device)
        .unsqueeze(0);

    let _g = tch::no_grad_guard();

    let header = wav_io::new_header(32000, 16, false, true);

    let ssl = gsv::SSL::new(&ssl_path, device).unwrap();
    let t2s = gsv::T2S::new(&t2s_path, device).unwrap();
    let vits = gsv::Vits::new(&vits_path, device).unwrap();

    let speaker = gsv::SpeakerV2Pro::new(&args.model, Arc::new(t2s), Arc::new(vits), Arc::new(ssl));

    let (prompts, refer, sv_emb) = speaker
        .pre_handle_ref(ref_audio_32k)
        .expect("Failed to pre handle ref");

    let st = std::time::Instant::now();
    let audio = speaker
        .infer(
            (
                prompts.shallow_clone(),
                refer.shallow_clone(),
                sv_emb.shallow_clone(),
            ),
            ref_seq.shallow_clone(),
            text_seq.shallow_clone(),
            ref_bert.shallow_clone(),
            text_bert.shallow_clone(),
            15,
        )
        .expect("failed to infer audio");
    info!("infer done, cost: {:?}", st.elapsed());

    let out_wav_name = Local::now().format("%Y-%m-%d-%H-%M-%S").to_string();
    let output = output_path.join(format!("{}.wav", out_wav_name));
    let audio_size = audio.size1().expect("Failed to get audio size") as usize;
    info!("audio size: {}", audio_size);
    let mut samples = vec![0f32; audio_size];
    audio
        .f_copy_data(&mut samples, audio_size)
        .expect("failed to copy audio data");
    info!("write output audio wav to {}", output.display());
    let mut file_out = fs::File::create(output).expect("failed to create output file");
    wav_io::write_to_file(&mut file_out, &header, &samples).expect("failed to write audio data");
}
