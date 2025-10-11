# nihility-gsv

基于：[gpt_sovits_rs](https://github.com/second-state/gpt_sovits_rs)重写，非常感谢这个库作者们的工作

使用tch在rust推理GPT-SoVITS模型，目前只支持cpu推理，13代i7可以实现`rtf < 0.5`

这个库是作为助手项目的语言合成模块设计的，程序的运行环境初始化不会集成在这个项目构建结果中。

# 使用方法

## 使用自动脚本安装

### Linux系统

```bash
curl -fsSL https://raw.githubusercontent.com/nihilityer/nihility-gsv/refs/heads/main/install.sh | bash
```

```bash
./nihility-gsv -t '心有所向，日复一日，必有精进。'
```

# TODO

- [ ] 流式推理
- [ ] 更多的模型推理方式选择
- [ ] 更高性能的模型推理
- [ ] 更多的模型支持