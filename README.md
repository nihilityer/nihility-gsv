# nihility-gsv

基于：[gpt_sovits_rs](https://github.com/second-state/gpt_sovits_rs)重写，非常感谢这位大佬的工作

使用tch在rust推理GPT-SoVITS模型

# 使用方法

**后续`lintorch`的设置以及模型下载将会自动化**

## 下载`lintorch`并设置环境变量

根据具体使用的后端下载并解压，设置环境变量：`LIBTORCH`指向解压后的地址（模型暂时基于cpu后端导出）

[linux版本地址](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip)

[win版本地址](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip)

## 模型下载

[huggingface](https://huggingface.co/nihilityer/nihility-gsv)

下载之后放在`model`目录即可

## 使用cli推理

```bash
nihility-gsv -t '心有所向，日复一日，必有精进。'
```

# TODO

- [ ] 自动初始化（下载依赖+默认模型）
- [ ] 推理api接口
- [ ] 推理GUI