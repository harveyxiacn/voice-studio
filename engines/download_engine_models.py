# -*- coding: utf-8 -*-
"""从 ModelScope 下载新引擎的模型到 engines/models（已下载完整的会跳过）。

    python download_engine_models.py            # 全部
    python download_engine_models.py VoxCPM2    # 只下载指定的
"""
import os
import shutil
import sys

from modelscope import snapshot_download

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# 名称: (ModelScope ID, 判断下载完整的文件)
REPOS = {
    "VoxCPM2": ("OpenBMB/VoxCPM2", "model.safetensors"),
    "Fun-CosyVoice3-0.5B": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "llm.pt"),
    "SenseVoiceSmall": ("iic/SenseVoiceSmall", "model.pt"),
    "Qwen3-TTS-12Hz-1.7B-Base": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors"),
}

for name, (repo, marker) in REPOS.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    target = os.path.join(MODELS, name)
    if os.path.exists(os.path.join(target, marker)):
        print("SKIP", name, "（已存在）", flush=True)
        continue
    shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
    for i in range(5):
        try:
            print("OK", name, snapshot_download(repo, local_dir=target), flush=True)
            break
        except Exception as e:
            print("retry", name, i, repr(e)[:300], flush=True)
