# -*- coding: utf-8 -*-
"""Fun-CosyVoice3 常驻工作进程。协议与 voxcpm_worker.py 相同（stdin JSON 行，@@RESULT@@ 输出）。

平静语气：zero_shot（参考录音 + 文稿，音色还原最好）
其他情感 / 粤语：instruct2（参考录音 + 文字指令控制情绪和方言）
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cosyvoice_src")
sys.path[:0] = [SRC, os.path.join(SRC, "third_party", "Matcha-TTS")]
MODEL_DIR = os.path.join(HERE, "models", "Fun-CosyVoice3-0.5B")
SYS = "You are a helpful assistant."

EMO_INSTR = {
    "开心": "开心愉快的语气", "兴奋": "非常兴奋激动的语气", "温柔": "温柔的语气",
    "悲伤": "悲伤难过的语气", "生气": "生气的语气", "惊讶": "惊讶的语气", "严肃": "严肃认真的语气",
}

model = None


def _offline_first():
    """wetext 等组件每次都会联网检查 ModelScope，网络抖动时会导致崩溃：优先用本地缓存。"""
    try:
        import modelscope
        import modelscope.hub.snapshot_download as sd
        orig = sd.snapshot_download

        def snapshot_download(model_id, *a, **k):
            try:
                return orig(model_id, *a, **dict(k, local_files_only=True))
            except Exception:
                return orig(model_id, *a, **k)

        sd.snapshot_download = snapshot_download
        modelscope.snapshot_download = snapshot_download
    except Exception:
        pass


_offline_first()


def reply(obj):
    sys.stdout.write("@@RESULT@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def vram():
    import torch
    return int(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0


def load(_lora=None):
    global model
    if model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        model = AutoModel(model_dir=MODEL_DIR, fp16=True)


def gen_one(it, speed, style, mode="auto"):
    """mode：auto（情感或粤语时用指令模式）/ zero_shot（参考录音+文稿，音色还原最好）/ instruct（指令模式）"""
    emo = it.get("emotion") if style else None
    yue = it.get("lang") == "yue"
    use_instr = mode == "instruct" or (mode == "auto" and ((emo in EMO_INSTR) or yue)) or not it.get("ref_text")
    if use_instr:
        req = []
        if yue:
            req.append("用广东话")
        if emo in EMO_INSTR:
            req.append(f"用{EMO_INSTR[emo]}")
        instr = f"{SYS} 请{'、'.join(req)}说这句话。<|endofprompt|>"
        gen = model.inference_instruct2(it["text"], instr, it["ref_wav"], stream=False, speed=speed)
    else:
        gen = model.inference_zero_shot(it["text"], f"{SYS}<|endofprompt|>{it.get('ref_text', '')}",
                                        it["ref_wav"], stream=False, speed=speed)
    return [out["tts_speech"] for out in gen]


def tts(cmd):
    import torch
    import torchaudio
    sr = model.sample_rate
    pause = torch.zeros(1, int(sr * float(cmd.get("pause", 0.3))))
    pieces = []
    for it in cmd["items"]:
        for w in gen_one(it, float(cmd.get("speed", 1.0)), cmd.get("style", True), cmd.get("mode", "auto")):
            pieces += [w.cpu(), pause]
    audio = torch.cat(pieces, dim=1)
    peak = float(audio.abs().max())
    if peak > 0.98:  # 防止爆音
        audio = audio * (0.95 / peak)
    torchaudio.save(cmd["out"], audio, sr)
    return {"ok": True, "out": cmd["out"], "dur": round(audio.shape[1] / sr, 2), "vram": vram()}


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
