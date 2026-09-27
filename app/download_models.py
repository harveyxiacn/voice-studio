# -*- coding: utf-8 -*-
"""下载粤语识别与情感识别模型（在 GPT-SoVITS 目录下用自带 runtime 运行）。"""
import os

from modelscope import snapshot_download

APP = os.path.dirname(os.path.abspath(__file__))
GSV = os.path.join(os.path.dirname(APP), "GPT-SoVITS")

yue = "speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
print("yue:", snapshot_download("iic/" + yue, local_dir=os.path.join(GSV, "tools", "asr", "models", yue)))
print("emotion:", snapshot_download("iic/emotion2vec_plus_base",
                                    local_dir=os.path.join(APP, "models", "emotion2vec_plus_base")))
