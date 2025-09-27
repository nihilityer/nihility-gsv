use nihility_gsv::{gsv, tch, text, text::G2PConfig};
use std::sync::Arc;
use time::format_description::well_known::Iso8601;
use tracing::info;
use tracing::metadata::LevelFilter;
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, fmt};

fn main() {
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

    let g2p_conf = G2PConfig::new("model/mini-bart-g2p.pt".to_string()).with_chinese(
        "model/g2pw_model.pt".to_string(),
        "model/bert_model.pt".to_string(),
    );

    let device = tch::Device::cuda_if_available();
    info!("device: {:?}", device);

    let g2p = g2p_conf.build(device).unwrap();

    let ref_text = "在二位牌局开始后不久。看到大家如此投入，我也不想打断。";
    let ref_path = "ref/ref.wav";

    let file = std::fs::File::open(ref_path).unwrap();
    let (head, mut ref_audio_samples) = wav_io::read_from_file(file).unwrap();
    info!("head: {:?}", head);
    if head.sample_rate != 32000 {
        info!("ref audio sample rate: {}, need 32000", head.sample_rate);
        ref_audio_samples = wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32000);
    }

    info!("load ht ref done");
    info!("start write file");

    let text = "心有所向，日复一日，必有精进。";
    let (text_seq, text_bert) = text::get_phone_and_bert(&g2p, text).unwrap();

    let (ref_seq, ref_bert) = text::get_phone_and_bert(&g2p, ref_text).unwrap();

    let ref_audio_32k = tch::Tensor::from_slice(&ref_audio_samples)
        .to_device(device)
        .unsqueeze(0);

    let _g = tch::no_grad_guard();

    let header = wav_io::new_header(32000, 16, false, true);

    let ssl = gsv::SSL::new("model/ssl_model.pt", device).unwrap();
    let t2s = gsv::T2S::new("model/t2s.cpu.pt", device).unwrap();
    let vits = gsv::Vits::new("model/vits.cpu.pt", device).unwrap();

    let speaker = gsv::SpeakerV2Pro::new("ht", Arc::new(t2s), Arc::new(vits), Arc::new(ssl));

    let (prompts, refer, sv_emb) = speaker.pre_handle_ref(ref_audio_32k).unwrap();

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
        .unwrap();
    info!("infer done, cost: {:?}", st.elapsed());

    let output = "out/out.wav";
    let audio_size = audio.size1().unwrap() as usize;
    println!("audio size: {}", audio_size);

    println!("start save audio {output}");
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    println!("start write file {output}");
    let mut file_out = std::fs::File::create(output).unwrap();
    wav_io::write_to_file(&mut file_out, &header, &samples).unwrap();
    info!("write file done");
}
