use nihility_gsv::{NihilityGsv, NihilityGsvConfig, NihilityGsvInferParam};
use nihility_rpc::common::{AudioData, ExecuteData, ExecuteRequest, ExecuteResponse};
use nihility_rpc::server::ExecuteServer;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio::spawn;
use tokio::sync::{Mutex, mpsc};
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::tokio_stream::{Stream, StreamExt};
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status, Streaming};
use tracing::{debug, error};

const CHANNEL_CAPACITY: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NihilityGsvRpcConfig {
    server_addr: String,
    server_port: u16,
}

type StreamResp = Pin<Box<dyn Stream<Item = Result<ExecuteResponse, Status>> + Send>>;

pub struct NihilityGsvRpcServer {
    gsv: Mutex<NihilityGsv>,
}

#[tonic::async_trait]
impl nihility_rpc::server::Execute for NihilityGsvRpcServer {
    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<ExecuteResponse>, Status> {
        debug!(?request, "execute");
        let req: ExecuteData = request
            .into_inner()
            .try_into()
            .map_err(|e| Status::invalid_argument(format!("{:?}", e)))?;
        debug!("Gsv Executing Request Data: {:?}", req);
        match req {
            ExecuteData::String(text) => {
                let audio_data = self
                    .gsv
                    .lock()
                    .await
                    .infer(NihilityGsvInferParam {
                        text,
                        ..Default::default()
                    })
                    .map_err(|e| Status::from_error(Box::new(e)))?;
                Ok(Response::new(
                    ExecuteData::Audio(AudioData {
                        data: audio_data,
                        ..Default::default()
                    })
                    .into(),
                ))
            }
            _ => Err(Status::new(
                Code::InvalidArgument,
                "Gsv Executing request must be a string",
            )),
        }
    }

    type ExecuteStreamOutStream = StreamResp;

    async fn execute_stream_out(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStreamOutStream>, Status> {
        debug!(?request, "execute_stream_out");
        Err(Status::unimplemented("stream out not supported"))
    }

    type ExecuteStreamStream = StreamResp;

    async fn execute_stream(
        &self,
        request: Request<Streaming<ExecuteRequest>>,
    ) -> Result<Response<Self::ExecuteStreamStream>, Status> {
        debug!(?request, "execute_stream");
        let mut req_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let gsv = self.gsv.lock().await.clone();
        spawn(async move {
            while let Some(req) = req_stream.next().await {
                match req {
                    Ok(ok_req) => {
                        let transfer_result: Result<ExecuteData, Status> = ok_req
                            .try_into()
                            .map_err(|e| Status::invalid_argument(format!("{:?}", e)));
                        debug!("execute_stream Executing request: {:?}", transfer_result);
                        match transfer_result {
                            Ok(req_data) => match req_data {
                                ExecuteData::String(text) => {
                                    match gsv.infer(NihilityGsvInferParam {
                                        text,
                                        ..Default::default()
                                    }) {
                                        Ok(data) => {
                                            match tx
                                                .send(Ok(ExecuteData::Audio(AudioData {
                                                    data,
                                                    ..Default::default()
                                                })
                                                .into()))
                                                .await
                                            {
                                                Ok(_) => {}
                                                Err(e) => {
                                                    error!("Send to channel fail: {}", e);
                                                }
                                            }
                                        }
                                        Err(e) => match tx
                                            .send(Err(Status::from_error(Box::new(e))))
                                            .await
                                        {
                                            Ok(_) => {}
                                            Err(e) => {
                                                error!("Send to channel fail: {}", e);
                                            }
                                        },
                                    }
                                }
                                _ => {
                                    error!(
                                        "Gsv Executing request must be a string: {:?}",
                                        req_data
                                    );
                                    match tx
                                        .send(Err(Status::invalid_argument(
                                            "Gsv Executing request must be a string",
                                        )))
                                        .await
                                    {
                                        Ok(_) => {}
                                        Err(e) => {
                                            error!("Send to channel fail: {}", e);
                                        }
                                    }
                                }
                            },
                            Err(status) => match tx.send(Err(status)).await {
                                Ok(_) => {}
                                Err(e) => {
                                    error!("Send to channel fail: {}", e);
                                }
                            },
                        }
                    }
                    Err(err_req) => {
                        error!("execute_stream Executing request error: {:?}", err_req);
                    }
                }
            }
        });
        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::ExecuteStreamOutStream
        ))
    }
}

#[tokio::main]
async fn main() {
    nihility_log::init().expect("could not init log");
    let gsv_rpc_config =
        nihility_config::get_config::<NihilityGsvRpcConfig>("nihility-gsv-rpc".to_string())
            .expect("could not get nihility gsv api config");
    let device = tch::Device::cuda_if_available();
    let gsv = nihility_config::get_config::<NihilityGsvConfig>(env!("CARGO_PKG_NAME").to_string())
        .expect("could not get inner config")
        .init(device)
        .expect("Failed to init gsv");
    Server::builder()
        .add_service(ExecuteServer::new(NihilityGsvRpcServer {
            gsv: Mutex::new(gsv),
        }))
        .serve(
            format!(
                "{}:{}",
                gsv_rpc_config.server_addr, gsv_rpc_config.server_port
            )
            .parse()
            .expect("Rpc Server Addr Config Error"),
        )
        .await
        .expect("Grpc Start Run Fail")
}

impl Default for NihilityGsvRpcConfig {
    fn default() -> Self {
        NihilityGsvRpcConfig {
            server_addr: "[::1]".to_string(),
            server_port: 8080,
        }
    }
}
