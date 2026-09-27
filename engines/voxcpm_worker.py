# -*- coding: utf-8 -*-
"""VoxCPM2 常驻工作进程。stdin 每行一个 JSON 命令，结果以 @@RESULT@@ 前缀输出一行 JSON。

命令：
  {"cmd": "load", "lora": null | "LoRA 目录"}
  {"cmd": "tts", "items": [{"text", "ref_wav", "ref_text", "emotion", "lang"}], "out": "x.wav",
   "pause": 0.3, "cfg": 2.0, "steps": 10, "style": true}
  {"cmd": "quit"}
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "VoxCPM2")

# 情感 -> 风格描述（VoxCPM2 的“可控克隆”：文本开头括号里写风格）
STYLE = {
    "开心": "开心愉快的语气", "兴奋": "兴奋激动的语气，语速稍快", "温柔": "温柔轻声的语气",
    "悲伤": "悲伤低落的语气，语速稍慢", "生气": "生气愤怒的语气", "惊讶": "惊讶的语气",
    "严肃": "严肃认真的语气",
}

model = None
lora_path = None


def reply(obj):
    sys.stdout.write("@@RESULT@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def vram():
    import torch
    return int(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0


def _create(lora):
    global model
    import gc
    import torch
    from voxcpm import VoxCPM
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, optimize=False, device="cuda",
                                   lora_weights_path=lora or None)


def load(lora):
    """按需加载/切换 LoRA。不带 LoRA 初始化的模型无法再挂 LoRA，此时重新加载模型。"""
    global model, lora_path
    if model is None:
        _create(lora)
    elif lora != lora_path:
        if lora:
            try:
                if lora_path:
                    model.unload_lora()
                model.load_lora(lora)
            except RuntimeError:
                _create(lora)
        elif lora_path:
            model.unload_lora()
    lora_path = lora


def build_args(it, style):
    """默认：极致克隆（同情感的参考录音 + 文稿续写），音色最像，情感来自所选参考录音。
    style=True（强化情感）：可控克隆（只给参考录音，文本开头括号写风格指令）。
    实测：可控克隆音色相似度明显下降（0.60 → 0.39），所以只作为可选项。
    注意：极致克隆模式下不能加括号指令，否则会被念出来。"""
    text = it["text"].strip()
    kw = {"reference_wav_path": it["ref_wav"]} if it.get("ref_wav") else {}
    ctrl = []
    if style and it.get("emotion") in STYLE:
        ctrl.append(STYLE[it["emotion"]])
    if ctrl:
        if it.get("lang") == "yue":
            ctrl.append("用粤语")
        return f"({'，'.join(ctrl)})" + text, kw
    if it.get("ref_wav") and it.get("ref_text"):
        kw.update(prompt_wav_path=it["ref_wav"], prompt_text=it["ref_text"])
    return text, kw


def tts(cmd):
    import numpy as np
    import soundfile as sf
    sr = model.tts_model.sample_rate
    pieces = []
    for i, it in enumerate(cmd["items"]):
        text, kw = build_args(it, cmd.get("style", True))
        if cmd.get("seed") is not None:
            kw["seed"] = int(cmd["seed"]) + i
        wav = model.generate(text=text, cfg_value=float(cmd.get("cfg", 2.0)),
                             inference_timesteps=int(cmd.get("steps", 10)), normalize=True, **kw)
        pieces += [np.asarray(wav, dtype=np.float32), np.zeros(int(sr * float(cmd.get("pause", 0.3))), np.float32)]
    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max()) if len(audio) else 0
    if peak > 0.98:  # 防止爆音
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
                load(cmd.get("lora"))
                res = {"ok": True, "vram": vram()}
            elif cmd["cmd"] == "tts":
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
