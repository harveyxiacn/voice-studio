# -*- coding: utf-8 -*-
"""人声分离：用 BS-Roformer 去掉背景音乐和环境噪音，可选再去混响。
GPT-SoVITS runtime、工作目录 GPT-SoVITS 运行：

    python separate.py <输出目录> <文件1> [文件2 ...] [--dereverb]

每个输入输出一行 @@OUT@@<人声wav路径>（顺序与输入一致）。
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

GSV = os.getcwd()
UVR = os.path.join(GSV, "tools", "uvr5")
sys.path[:0] = [GSV, UVR, os.path.join(GSV, "tools")]
WEIGHTS = os.path.join(UVR, "uvr5_weights")
ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755"
FFMPEG = os.path.join(GSV, "runtime", "ffmpeg.exe")


def to_44k_stereo(src, dst):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-vn", "-acodec", "pcm_s16le", "-ac", "2", "-ar", "44100", dst],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--dereverb", action="store_true")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    half = dev == "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = os.path.join(a.out_dir, "_tmp")
    from bsroformer import Roformer_Loader
    sep = Roformer_Loader(model_path=os.path.join(WEIGHTS, ROFORMER + ".ckpt"),
                          config_path=os.path.join(WEIGHTS, ROFORMER + ".yaml"), device=dev, is_half=half)
    results = []
    for i, src in enumerate(a.inputs):
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        base = f"in{i:02d}"
        wav = os.path.join(tmp, base + ".wav")
        print(f"[sep] {os.path.basename(src)}：分离人声…", flush=True)
        to_44k_stereo(src, wav)
        sep._path_audio_(wav, os.path.join(tmp, "other"), os.path.join(tmp, "vocal"), "wav")
        vocal = (glob.glob(os.path.join(tmp, "vocal", "*vocal*.wav")) or glob.glob(os.path.join(tmp, "vocal", "*.wav")))[0]
        results.append(vocal)
        stem = os.path.splitext(os.path.basename(src))[0]
        final = os.path.join(a.out_dir, f"{stem}_人声.wav")
        shutil.move(vocal, final)
        results[-1] = final
    del sep
    torch.cuda.empty_cache()
    if a.dereverb:
        from vr import AudioPreDeEcho
        de = AudioPreDeEcho(agg=10, model_path=os.path.join(WEIGHTS, "VR-DeEchoNormal.pth"), device=dev, is_half=half)
        for k, v in enumerate(results):
            print(f"[sep] {os.path.basename(v)}：去混响…", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            # DeEcho 模型的输出目录是反的：干声写进 ins_root（文件名 vocal_*），混响写进 vocal_root
            de._path_audio_(v, vocal_root=os.path.join(tmp, "echo"), ins_root=os.path.join(tmp, "dry"), format="wav")
            dry = glob.glob(os.path.join(tmp, "dry", "vocal_*.wav")) or glob.glob(os.path.join(tmp, "dry", "*.wav"))
            if dry:
                final = v[:-4] + "_去混响.wav"
                # VR 输出为 44.1k 立体声，统一转成单声道
                subprocess.run([FFMPEG, "-v", "error", "-y", "-i", dry[0], "-ac", "1", final], check=True)
                results[k] = final
    shutil.rmtree(tmp, ignore_errors=True)
    for r in results:
        print("@@OUT@@" + r, flush=True)


if __name__ == "__main__":
    main()
