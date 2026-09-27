# -*- coding: utf-8 -*-
"""引擎管理：每个引擎一个常驻工作进程（网页主进程不占显存），同一时间只保留一个引擎占显存。
对外提供逐句生成 synth_sentence()，整段/字幕/重做见 takes.py。"""
import json
import os
import random
import subprocess
import threading
import time

import core
import gpu
from tts_engine import Engine, TEXT_LANGS

ENG_DIR = os.path.join(core.ROOT, "engines")
WORKERS = {
    "VoxCPM2": (os.path.join(ENG_DIR, "voxcpm_env", "Scripts", "python.exe"), os.path.join(ENG_DIR, "voxcpm_worker.py")),
    "CosyVoice3": (os.path.join(ENG_DIR, "cosyvoice_env", "Scripts", "python.exe"), os.path.join(ENG_DIR, "cosyvoice_worker.py")),
    "GPT-SoVITS": (core.PY, os.path.join(core.APP_DIR, "gsv_worker.py")),
    "Qwen3-TTS": (os.path.join(ENG_DIR, "qwen_env", "Scripts", "python.exe"), os.path.join(ENG_DIR, "qwen_worker.py")),
}
ENGINE_NAMES = ["VoxCPM2", "CosyVoice3", "Qwen3-TTS", "GPT-SoVITS"]
ENGINE_DESC = {
    "VoxCPM2": "最像本人，普通话/粤语都推荐；微调后效果最好",
    "CosyVoice3": "普通话自然流畅；粤语口音不地道（不推荐）",
    "GPT-SoVITS": "省显存；需训练后才稳定，零样本粤语易漏字",
    "Qwen3-TTS": "普通话音色语调像；粤语口音不地道（不推荐）",
}
MODEL_FILES = {
    "VoxCPM2": os.path.join(ENG_DIR, "models", "VoxCPM2", "model.safetensors"),
    "CosyVoice3": os.path.join(ENG_DIR, "models", "Fun-CosyVoice3-0.5B", "llm.pt"),
    "Qwen3-TTS": os.path.join(ENG_DIR, "models", "Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors"),
}


def available():
    return [n for n in ENGINE_NAMES
            if n == "GPT-SoVITS" or (all(os.path.exists(p) for p in WORKERS[n]) and os.path.exists(MODEL_FILES[n]))]


def worker_cwd(name):
    return core.GSV if name == "GPT-SoVITS" else ENG_DIR


class Worker:
    def __init__(self, name):
        self.name = name
        self.p = None
        self.lock = threading.Lock()
        self.logs = []
        self.lora = "?"

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def start(self):
        py, script = WORKERS[self.name]
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false"})
        env["PATH"] = os.path.join(core.GSV, "runtime") + os.pathsep + env.get("PATH", "")
        self.p = subprocess.Popen([py, script], cwd=worker_cwd(self.name), env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lora = "?"

    def call(self, cmd, timeout=1800):
        with self.lock:
            if not self.alive():
                self.start()
            self.p.stdin.write((json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
            self.p.stdin.flush()
            t0 = time.time()
            for raw in iter(self.p.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if "@@RESULT@@" in line:  # 结果标记可能跟在进度条（\r 刷新）后面
                    return json.loads(line.split("@@RESULT@@", 1)[1])
                if line:
                    self.logs = (self.logs + [line])[-200:]
                if time.time() - t0 > timeout:
                    break
            tail = "\n".join(self.logs[-8:])
            self.stop()
            raise RuntimeError(f"{self.name} 进程异常退出：\n{tail}")

    def stop(self):
        if self.p is None:
            return
        try:
            if self.alive():
                self.p.stdin.write(b'{"cmd":"quit"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
        except Exception:
            pass
        if self.alive():
            subprocess.call(f"taskkill /t /f /pid {self.p.pid}", shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.p = None
        self.lora = "?"  # 进程已关，下次必须重新加载模型


_workers = {n: Worker(n) for n in WORKERS}


def loaded():
    """当前已加载（占显存）的引擎名列表。"""
    return [n for n, w in _workers.items() if w.alive()]


def release_all(keep=None):
    """释放显存：停掉其他引擎进程，只保留 keep。"""
    for n, w in _workers.items():
        if n != keep:
            w.stop()


def prepare(engine, voice, use_lora=True):
    """切换到 engine：释放其他引擎、检查显存、加载模型（VoxCPM2 按需加载该语音包的 LoRA）。"""
    release_all(keep=engine)
    w = _workers[engine]
    gpu.check_free(engine, already_loaded=w.alive())
    lora = voice.get("voxcpm_lora") if (engine == "VoxCPM2" and use_lora) else None
    if engine != "GPT-SoVITS" and (w.lora != lora or not w.alive()):
        r = w.call({"cmd": "load", "lora": lora})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        w.lora = lora
    return w


def synth_sentence(engine, voice, emo, text, p, seed, out, use_lora=True):
    """生成一句话到 out。p: 参数字典（lang_label, expr, speed, multi, strong）。
    返回 (用到的参考情感, 是否命中情感, 参考录音文件名或 None)。text 应是已规范化的「朗读文字」。"""
    w = prepare(engine, voice, use_lora)
    text_lang = TEXT_LANGS[p["lang_label"]]
    if engine == "GPT-SoVITS":
        r = w.call({"cmd": "gsv", "out": out, "args": [voice["id"], f"[{emo}]{text}", p["lang_label"], emo, False,
                                                       p["expr"], p["speed"], 0.05, seed, p["multi"]]})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        hit = "无此情感素材" not in r.get("report", "")
        return (emo if hit else "平静"), hit, None
    ref, _aux, hit = Engine.pick_refs(voice, emo, text_lang, random.Random(seed))
    x = float(p["expr"])
    r = w.call({"cmd": "tts", "out": out, "pause": 0.0, "speed": float(p["speed"]), "seed": seed, "expr": x,
                "style": bool(p.get("strong")) if engine == "VoxCPM2" else x > 0.05,
                "cfg": round(1.5 + 1.5 * (1 - x), 2), "steps": 10,
                "items": [{"text": text, "emotion": emo, "lang": "yue" if "yue" in text_lang else "zh",
                           "ref_wav": os.path.join(voice["dir"], "refs", ref["file"]), "ref_text": ref["text"]}]})
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    return ref["emotion"], hit, ref["file"]
