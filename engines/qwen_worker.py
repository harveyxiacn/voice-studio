# -*- coding: utf-8 -*-
"""Qwen3-TTS（1.7B Base，声音克隆）常驻工作进程。协议与 voxcpm_worker.py 相同。

克隆模式：参考录音 + 文稿（ICL，音色还原最好）；没有文稿时只用声纹（x-vector）。
注意：官方克隆模型支持的语言里没有粤语，粤语文本按「中文」处理，效果以实测为准。
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "Qwen3-TTS-12Hz-1.7B-Base")
model = None


def reply(obj):
    sys.stdout.write("@@RESULT@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def vram():
    import torch
    return int(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0


def load():
    global model
    if model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16,
                                              attn_implementation="sdpa")


def tts(cmd):
    import numpy as np
    import soundfile as sf
    import torch
    pieces, sr = [], 24000
    for i, it in enumerate(cmd["items"]):
        if cmd.get("seed") is not None:
            torch.manual_seed(int(cmd["seed"]) + i)
        x = float(cmd.get("expr", 0.6))
        kw = dict(do_sample=True, temperature=round(0.6 + 0.5 * x, 2), top_p=0.9, repetition_penalty=1.05)
        has_text = bool(it.get("ref_text"))
        wavs, sr = model.generate_voice_clone(text=it["text"], language="Chinese", ref_audio=it["ref_wav"],
                                              ref_text=it.get("ref_text") if has_text else None,
                                              x_vector_only_mode=not has_text, **kw)
        pieces += [np.asarray(wavs[0], dtype=np.float32), np.zeros(int(sr * float(cmd.get("pause", 0.3))), np.float32)]
    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max()) if len(audio) else 0
    if peak > 0.98:
        audio = audio * (0.95 / peak)
    sf.write(cmd["out"], audio, sr)
    return {"ok": True, "out": cmd["out"], "dur": round(len(audio) / sr, 2), "vram": vram()}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            cmd = json.loads(line)
            t0 = time.time()
            if cmd["cmd"] == "quit":
                reply({"ok": True})
                break
            if cmd["cmd"] == "load":
                load()
                res = {"ok": True, "vram": vram()}
            elif cmd["cmd"] == "tts":
                load()
                res = tts(cmd)
            else:
                res = {"ok": False, "error": "unknown cmd"}
            res["sec"] = round(time.time() - t0, 1)
            reply(res)
        except Exception as e:
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
