# 声音工坊 · 一键安装（Windows + NVIDIA 显卡）
# 每一步都会先检查是否已完成，重复运行是安全的（已装好的会跳过）。
# 用法：右键「使用 PowerShell 运行」，或在本目录执行：powershell -ExecutionPolicy Bypass -File install.ps1
# 可选参数：-SkipQwen -SkipCosyVoice -SkipVoxCPM（跳过对应引擎）

param([switch]$SkipQwen, [switch]$SkipCosyVoice, [switch]$SkipVoxCPM)

# 原生程序（uv/git/python）会把进度和警告写到 stderr，PowerShell 5.1 在 Stop 模式下会误判为错误；
# 这里用 Continue，改为逐个检查 $LASTEXITCODE
$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
$Eng = Join-Path $Root "engines"
$Gsv = Join-Path $Root "GPT-SoVITS"
$GsvPy = Join-Path $Gsv "runtime\python.exe"
$env:PYTHONIOENCODING = "utf-8"
# 国内镜像（也可以改成官方源）
$PyPI = "https://mirrors.aliyun.com/pypi/simple/"
$TorchCu124 = "https://mirrors.aliyun.com/pytorch-wheels/cu124/"
$TorchCu121 = "https://mirrors.aliyun.com/pytorch-wheels/cu121/"
$HfMirror = "https://hf-mirror.com"
$GsvPkg = "GPT-SoVITS-v2pro-20250604.7z"

function Step($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Done($msg) { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "缺少 $cmd：$hint" }
}

Step "0. 检查环境"
Need "nvidia-smi" "需要 NVIDIA 显卡和驱动（建议显存 ≥ 8GB）"
Need "git" "请先安装 Git：https://git-scm.com/download/win"
Need "curl.exe" "Windows 10 以上自带 curl"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  未找到 uv，正在安装…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    Need "uv" "请手动安装 uv：https://docs.astral.sh/uv/"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
Done "环境检查通过"

Step "1. GPT-SoVITS（整合包，含 Python 运行环境、切片/识别/人声分离工具）"
if (Test-Path $GsvPy) {
    Done "已安装，跳过"
} else {
    $dl = Join-Path $Root "downloads"
    New-Item -ItemType Directory -Force $dl | Out-Null
    $pkg = Join-Path $dl $GsvPkg
    $seven = Join-Path $dl "7zr.exe"
    if (-not (Test-Path $seven)) { curl.exe -L -o $seven https://www.7-zip.org/a/7zr.exe }
    Write-Host "  下载整合包（约 8GB，可断点续传）…"
    curl.exe -L -C - --retry 5 -o $pkg "$HfMirror/lj1995/GPT-SoVITS-windows-package/resolve/main/$GsvPkg"
    if ($LASTEXITCODE -ne 0) { throw "下载失败，请重试（会从断点继续）" }
    & $seven x -y "-o$dl\_extract" $pkg | Out-Null
    Move-Item (Join-Path $dl "_extract\GPT-SoVITS-v2pro-20250604") $Gsv
    Remove-Item (Join-Path $dl "_extract") -Recurse -Force
    Done "GPT-SoVITS 已安装（确认可用后可以删除 downloads 文件夹）"
}
Write-Host "  下载粤语识别与情感识别模型…"
Push-Location $Gsv
& $GsvPy (Join-Path $Root "app\download_models.py")
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "识别模型下载失败，请重试" }
Done "识别模型就绪"

function New-Env($name, $pyver) {
    $py = Join-Path $Eng "$name\Scripts\python.exe"
    if (-not (Test-Path $py)) { uv venv (Join-Path $Eng $name) --python $pyver }
    return $py
}

function Uv-Install($envName, [string[]]$pkgArgs) {
    $env:VIRTUAL_ENV = Join-Path $Eng $envName
    uv pip install @pkgArgs
    if ($LASTEXITCODE -ne 0) { throw "安装失败：$envName $pkgArgs" }
    Remove-Item Env:VIRTUAL_ENV
}

function Test-Import($py, $code) {
    & $py -W ignore -c $code *> $null
    return $LASTEXITCODE -eq 0
}

New-Item -ItemType Directory -Force $Eng | Out-Null

if (-not $SkipVoxCPM) {
    Step "2. VoxCPM2 引擎（Python 3.11 + PyTorch 2.6 CUDA）"
    $src = Join-Path $Eng "voxcpm_src"
    if (-not (Test-Path $src)) { git clone https://github.com/OpenBMB/VoxCPM.git $src; if ($LASTEXITCODE -ne 0) { throw "克隆 VoxCPM 失败（需要能访问 GitHub）" } }
    $py = New-Env "voxcpm_env" "3.11"
    if (Test-Import $py "import torch, voxcpm; assert torch.cuda.is_available()") {
        Done "已安装，跳过"
    } else {
        Uv-Install "voxcpm_env" @("torch==2.6.0+cu124", "torchaudio==2.6.0+cu124", "--index-url", $PyPI, "--find-links", $TorchCu124, "--index-strategy", "unsafe-best-match")
        Uv-Install "voxcpm_env" @("-e", $src, "--index-url", $PyPI)
        Done "VoxCPM2 环境已安装"
    }
}

if (-not $SkipCosyVoice) {
    Step "3. CosyVoice3 引擎（Python 3.10 + PyTorch 2.3.1 CUDA）"
    $src = Join-Path $Eng "cosyvoice_src"
    if (-not (Test-Path $src)) { git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $src; if ($LASTEXITCODE -ne 0) { throw "克隆 CosyVoice 失败（需要能访问 GitHub）" } }
    $py = New-Env "cosyvoice_env" "3.10"
    if (Test-Import $py "import torch, hyperpyyaml, pkg_resources; assert torch.cuda.is_available()") {
        Done "已安装，跳过"
    } else {
        # 去掉 requirements 里的境外额外源，统一走国内镜像
        $req = Join-Path $Eng "_cosy_req.txt"
        Get-Content (Join-Path $src "requirements.txt") | Where-Object { $_ -notmatch "^--" } | Set-Content -Encoding utf8 $req
        Uv-Install "cosyvoice_env" @("-r", $req, "--index-url", $PyPI, "--find-links", $TorchCu121, "--index-strategy", "unsafe-best-match", "--build-constraint", (Join-Path $Eng "build_constraints.txt"))
        Uv-Install "cosyvoice_env" @("setuptools<70", "--index-url", $PyPI)
        Remove-Item $req
        Done "CosyVoice3 环境已安装"
    }
}

if (-not $SkipQwen) {
    Step "4. Qwen3-TTS 引擎（Python 3.12）"
    $py = New-Env "qwen_env" "3.12"
    if (Test-Import $py "import torch, qwen_tts; assert torch.cuda.is_available()") {
        Done "已安装，跳过"
    } else {
        Uv-Install "qwen_env" @("qwen-tts", "--index-url", $PyPI)
        # qwen-tts 可能把 PyTorch 换成 CPU 版，这里强制装回 CUDA 版
        Uv-Install "qwen_env" @("--reinstall", "torch==2.6.0+cu124", "torchaudio==2.6.0+cu124", "--index-url", $PyPI, "--find-links", $TorchCu124, "--index-strategy", "unsafe-best-match")
        Done "Qwen3-TTS 环境已安装"
    }
}

Step "5. 下载模型（ModelScope，已存在的会跳过）"
$dlPy = Join-Path $Eng "voxcpm_env\Scripts\python.exe"
if (-not (Test-Path $dlPy)) { $dlPy = $GsvPy }
$names = @("SenseVoiceSmall")
if (-not $SkipVoxCPM) { $names += "VoxCPM2" }
if (-not $SkipCosyVoice) { $names += "Fun-CosyVoice3-0.5B" }
if (-not $SkipQwen) { $names += "Qwen3-TTS-12Hz-1.7B-Base" }
& $dlPy (Join-Path $Eng "download_engine_models.py") @names
if ($LASTEXITCODE -ne 0) { throw "模型下载失败，请重试（已下载的不会重复下载）" }
Done "模型就绪"

Step "安装完成"
Write-Host "双击「启动声音工坊.bat」打开网页：http://127.0.0.1:7866" -ForegroundColor Green
