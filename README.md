# 🎙️ 声音工坊 Voice Studio

本地运行的**声音克隆工作台**，支持普通话和粤语，面向中文短视频（剪映）创作者：上传 vlog 或录音，自动分出每个说话人、切片、识别文字；校对后生成**语音包**，输入文案就能用这个声音说话，同时导出 **SRT 字幕**。

- 完全本地运行，素材和声音不上传任何服务器
- 面向 8GB 显存的 NVIDIA 显卡设计（在 RTX 3070 Laptop 8GB 上开发和实测）
- Gradio 网页界面 + 命令行训练

> ⚠️ 只能克隆你本人或已获得明确授权的声音。请勿用于冒充他人、诈骗或任何违法用途。

## 功能

| 模块 | 说明 |
|---|---|
| 导入素材 | 上传录音或视频（可多选）；可选先做 **BS-Roformer 人声分离**，去掉背景音乐和噪音；**自动识别说话人**（单人/多人），多个文件会联合识别；根据**声纹自动认出**已有的声音 |
| 按句切分 | 用 SenseVoice 逐字时间戳，只在句子边界下刀，不会切到半个字；可以提供准确文稿，自动逐字对齐 |
| 双识别校对 | 两个识别模型交叉比对，结果不一致的标为「疑点」并排在前面；支持快捷键；分错人的片段可以移到其他声音 |
| 录音向导 | 普通话、粤语各 48 句台词（8 种情绪）；点「开始录音」才会使用麦克风，停止后立即释放；自动检查太小声、爆音、噪音 |
| 四个引擎 | **VoxCPM2**（支持 LoRA 微调）、**CosyVoice3**、**Qwen3-TTS**、**GPT-SoVITS**（支持训练）；同一时间只占用一个引擎的显存 |
| 文字转语音 | 逐句生成，情感可用 `[开心]` 标签指定，也可以自动判断；导出整段 wav 和 SRT 字幕；可单句重做；保留生成历史；多引擎对比试听 |
| 试听 | 在网页里批量播放和对比，1–5 星评分并记录备注 |
| 管理 | 改名、合并、删除声音；导出/导入语音包；磁盘清理；显卡互斥锁（网页和命令行共用）；显存预检 |

详细说明见 [使用说明.md](使用说明.md)，架构和各模块实测数据见 [设计说明.md](设计说明.md)。

## 安装（Windows 10/11 + NVIDIA 显卡）

需要：NVIDIA 显卡（建议 ≥8GB 显存）、[Git](https://git-scm.com/download/win)、约 **40GB** 磁盘空间。脚本会自动安装 [uv](https://docs.astral.sh/uv/)。

```powershell
git clone <本仓库地址> voice_clone
cd voice_clone
双击「安装.bat」    # 或：powershell -ExecutionPolicy Bypass -File install.ps1
```

安装脚本会依次完成下面几步。每一步都会先检查，已完成的会跳过，中断后重新运行会接着装：

1. 下载 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Windows 整合包（v2pro-20250604，约 8GB），自带 Python 运行环境、切片、识别和人声分离工具
2. 用 uv 创建三个独立的 Python 环境。它们依赖的 PyTorch 版本互相冲突，所以必须分开：
   - `engines/voxcpm_env`：Python 3.11，PyTorch 2.6 + CUDA 12.4
   - `engines/cosyvoice_env`：Python 3.10，PyTorch 2.3.1 + CUDA 12.1
   - `engines/qwen_env`：Python 3.12，PyTorch 2.6 + CUDA 12.4
3. 从 ModelScope 下载模型：VoxCPM2、Fun-CosyVoice3-0.5B、Qwen3-TTS-1.7B-Base、SenseVoiceSmall、emotion2vec、粤语 UniASR

默认使用国内镜像：阿里云 PyPI、阿里云 PyTorch 镜像、ModelScope、hf-mirror。不需要某个引擎时，可以用 `-SkipQwen`、`-SkipCosyVoice`、`-SkipVoxCPM` 跳过。

> 说明：安装脚本在已装好的环境里验证过「检查并跳过」的逻辑，但全新机器上的完整安装流程还没有端到端测试过。遇到问题欢迎提 Issue。

安装完成后，双击 **`启动声音工坊.bat`**，浏览器会打开 http://127.0.0.1:7866 。长时间训练可以用 **`训练.bat`**，在命令行里运行，关掉网页也不会中断。

## 显存实测（RTX 3070 Laptop 8GB）

| 任务 | 显存峰值 |
|---|---|
| VoxCPM2 推理 / LoRA 微调 | 约 6.9GB / 约 7.75GB（冻结的底座权重转为 bf16 后才能放进 8GB） |
| CosyVoice3 推理 | 约 4.3GB |
| Qwen3-TTS 1.7B 推理 | 约 4.6GB |
| GPT-SoVITS 推理 | 约 1.8GB |

训练前请关闭剪映等占用显存的软件。显存不够时，状态栏会提示是哪个程序占着显卡。

## 目录结构

```
app/        网页（voice_studio.py）、命令行（cli.py）、核心流程（core.py）、说话人识别、人声分离、录音向导、作品/字幕等
engines/    各引擎的工作进程与训练脚本（运行环境和模型由安装脚本下载，不在仓库里）
install.ps1 / 安装.bat / 启动声音工坊.bat / 训练.bat
```

运行时会生成这些目录，都**不会**提交到仓库（`.gitignore` 采用白名单方式，只放行代码和文档）：`GPT-SoVITS/`、`engines/*_env`、`engines/models`、`workspace/`（素材和切片）、`voices/`（语音包）、`outputs/`（生成结果）、`test-voice/`。

## 第三方项目与许可证

本仓库只包含声音工坊自己的代码。运行时会下载并调用以下项目和模型，**请分别遵守它们各自的许可证和使用条款**：

| 项目 | 用途 |
|---|---|
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT） | 引擎之一；切片、FunASR 识别、UVR5 / BS-Roformer 人声分离、ERes2NetV2 声纹 |
| [VoxCPM](https://github.com/OpenBMB/VoxCPM)（Apache-2.0） | VoxCPM2 引擎与 LoRA 微调 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（Apache-2.0） | CosyVoice3 引擎 |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0） | Qwen3-TTS 引擎 |
| [FunASR / SenseVoice / emotion2vec](https://github.com/modelscope/FunASR) | 语音识别、逐字时间戳、情感识别 |
| [Gradio](https://github.com/gradio-app/gradio)（Apache-2.0） | 网页界面 |

## 许可证

声音工坊自己的代码采用 [MIT License](LICENSE)。
