use chrono::Local;
use nihility_rpc::client::ExecuteClient;
use nihility_rpc::common::ExecuteData;
use std::fs;
use std::path::PathBuf;
use tonic::Request;
use tracing::{error, info};

#[tokio::test]
async fn test_rpc_client() {
    nihility_log::init().expect("could not init log");
    let addr = format!("http://[::1]:8080");
    let mut client = ExecuteClient::connect(addr).await.expect("connect error");
    info!("test service execute function");
    for _ in 0..3 {
        let execute_resp: ExecuteData = client
            .execute(Request::new(
                ExecuteData::String("心有所向，日复一日，必有精进。".to_string()).into(),
            ))
            .await
            .expect("execute response error")
            .into_inner()
            .try_into()
            .expect("execute response format error");
        match execute_resp {
            ExecuteData::Audio(audio_data) => {
                let wav_header = wav_io::new_header(
                    audio_data.sample_rate,
                    audio_data.sample_size as u16,
                    false,
                    audio_data.channels == 1,
                );
                let out_wav_name = Local::now().format("%Y-%m-%d-%H-%M-%S").to_string();
                let output_path = PathBuf::from("output");
                let output = output_path.join(format!("{}.wav", out_wav_name));
                let mut file_out = fs::File::create(&output).expect("Could not create output file");
                wav_io::write_to_file(&mut file_out, &wav_header, &audio_data.data)
                    .expect("write to file");
                info!("write output audio wav to {}", output.display());
            }
            _ => {
                error!("wrong execute response");
            }
        }
    }
}
