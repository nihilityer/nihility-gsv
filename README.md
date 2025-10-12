# nihility-gsv

基于：[gpt_sovits_rs](https://github.com/second-state/gpt_sovits_rs)重写，非常感谢这个库作者们的工作

使用tch在rust推理GPT-SoVITS模型，目前只支持cpu推理，13代i7可以实现`rtf < 0.5`

说明：

- 这个库是作为助手项目的语言合成模块设计的，程序的运行环境初始化不会集成在这个项目构建结果中。可以使用仓库中自动安装脚本进行安装。

- 目前只支持v2ProPlus，其他版本暂未导出。可以通过官方提供的导出脚本自行导出。

- 导出时注意选择`device`为`CPU`，如果需要支持cuda，需要自行下载`cuda`版本的`libtorch`

# 使用方法

## 安装

### Linux系统

```bash
curl -fsSL https://raw.githubusercontent.com/nihilityer/nihility-gsv/refs/heads/main/install.sh | bash
```

### windows系统

1. 下载一键安装脚本到需要安装的目录（自动下载文件较多，请注意不要直接放在桌面）：[下载地址](https://raw.githubusercontent.com/nihilityer/nihility-gsv/refs/heads/main/install.bat)
2. 双击执行

## 命令行推理

**注**：所有推理结果都在`output`目录下，根据生成完成的时间来设置文件名（默认生成在：`output`）

### Linux系统

```bash
./nihility-gsv-cli -t '心有所向，日复一日，必有精进。'
```

### windows系统

文件管理器中打开安装目录，在地址栏中输入`cmd`打开命令提示符，然后输入命令：

也支持直接双击执行

```cmd
nihility-gsv-cli.exe -t 心有所向，日复一日，必有精进。
```

## Api服务器

### Linux系统

```bash
./nihility-gsv-api
```

### Windows系统

双击`nihility-gsv-api.exe`执行

# 配置

有关gsv核心的模型配置文件默认为：`config/nihility-gsv.toml`，支持Json格式配置。

Api服务器相关配置文件默认为：`config/nihility-gsv-api.toml`，支持Json格式配置。

# TODO

- [x] API调用
- [ ] 流式推理
- [ ] 更多的模型推理方式选择
- [ ] 更高性能的模型推理
- [ ] 更多的模型支持