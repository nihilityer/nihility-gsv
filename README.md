# nihility-gsv

基于：[gpt_sovits_rs](https://github.com/second-state/gpt_sovits_rs)重写，非常感谢这个库作者们的工作

使用tch在rust推理GPT-SoVITS模型，目前只支持cpu推理，13代i7可以实现`rtf < 1`

这个库是作为助手项目的语言合成模块设计的，程序的运行环境初始化不会集成在这个项目构建结果中。

# 使用方法

## 下载`lintorch`并设置动态依赖库目录

- linux：将压缩包中lib目录中所有文件放入程序所在目录的lib目录
- windows： 将压缩包lib目录下所有文件放入程序所在目录

[linux版本地址](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip)

[win版本地址](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip)

## 模型下载

[huggingface](https://huggingface.co/nihilityer/nihility-gsv)

`base`文件夹下所有模型放在程序所在目录的base目录下，`default`目录下所有模型放在`model/default`目录下。

## 使用cli推理

```bash
nihility-gsv -t '心有所向，日复一日，必有精进。'
```

# TODO

- [ ] 流式推理
- [ ] 更多的模型推理方式选择
- [ ] 更高性能的模型推理
- [ ] 更多的模型支持