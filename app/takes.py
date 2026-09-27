# -*- coding: utf-8 -*-
"""作品（take）：逐句生成 → 拼成整段 wav + SRT 字幕；可单句重做；保留生成历史。

目录：outputs/_takes/<作品ID>/meta.json + s001.wav…；导出文件：outputs/<声音>_<引擎>_<时间>.wav/.srt
"""
import json
import os
import random
import re
import shutil
import time

import numpy as np

import core
import engines as ENG
from tts_engine import split_sentences

TAKES = os.path.join(core.OUTPUTS, "_takes")
os.makedirs(TAKES, exist_ok=True)
SUB_MAX = 16  # 每行字幕最多字数（剪映竖屏/横屏都比较合适）
SUB_SPLIT = re.compile(r"(?<=[，,、；;：:。！？!?…])")
TRAIL_PUNCT = "，,、；;：:。…"


def _dir(tid):
    return os.path.join(TAKES, tid)


def load(tid):
    with open(os.path.join(_dir(tid), "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(meta):
    with open(os.path.join(_dir(meta["id"]), "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def history(limit=50):
    """[(显示名, 作品ID)]，最新在前。"""
    out = []
    for tid in sorted(os.listdir(TAKES), reverse=True)[:limit]:
        p = os.path.join(TAKES, tid, "meta.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            first = m["sentences"][0]["text"][:14] if m["sentences"] else ""
            out.append((f"{m['created'][5:]} · {m['voice_name']} · {m['engine']} · {first}…", tid))
    return out


def _read(path):
    from scipy.io import wavfile
    sr, y = wavfile.read(path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2 ** 31
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(1)
    return sr, y


def _ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sub_lines(text):
    """把一句话按标点拆成不超过 SUB_MAX 字的字幕行，并去掉行尾标点。"""
    parts = [p for p in SUB_SPLIT.split(text.strip()) if p.strip()]
    lines, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > SUB_MAX:
            lines.append(buf)
            buf = ""
        buf += p
        while len(buf) > SUB_MAX * 1.5:  # 超长且没有标点：硬切
            lines.append(buf[:SUB_MAX])
            buf = buf[SUB_MAX:]
    if buf:
        lines.append(buf)
    # 剪映字幕习惯：去掉行尾标点，行内逗号顿号换成空格
    return [re.sub(r"[，,、；;：:]\s*", " ", l.strip().rstrip(TRAIL_PUNCT)).strip() or l.strip() for l in lines]


def _weight(s):
    return max(1, len(re.sub(r"[^\w]", "", s)))


def build(meta):
    """把各句拼成整段 wav，并按每句实际时长生成 SRT。"""
    d = _dir(meta["id"])
    pieces, srt, t, sr_all, n = [], [], 0.0, None, 0
    pause = float(meta["params"]["pause"])
    for i, s in enumerate(meta["sentences"]):
        sr, y = _read(os.path.join(d, s["file"]))
        if sr_all is None:
            sr_all = sr
        s["dur"] = round(len(y) / sr, 2)
        s["start"] = round(t, 2)
        lines = _sub_lines(s["text"])
        tot = sum(_weight(l) for l in lines)
        lt = t
        for l in lines:
            ld = s["dur"] * _weight(l) / tot
            n += 1
            srt.append(f"{n}\n{_ts(lt)} --> {_ts(lt + ld)}\n{l}\n")
            lt += ld
        pieces.append(y)
        t += s["dur"]
        if i < len(meta["sentences"]) - 1:
            pieces.append(np.zeros(int(sr * pause), np.float32))
            t += pause
    audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    peak = float(np.abs(audio).max()) if len(audio) else 0
    if peak > 0.98:
        audio = audio * (0.95 / peak)
    from scipy.io import wavfile
    wavfile.write(meta["wav"], sr_all or 32000, (audio * 32767).astype(np.int16))
    with open(meta["srt"], "w", encoding="utf-8-sig") as f:
        f.write("\n".join(srt))
    meta["total"] = round(t, 2)
    _save(meta)
    return meta


def create(engine, vid, text, params, log=lambda m: None):
    """params: lang_label, default_emo, auto_emo, expr, speed, pause, seed, multi, strong, use_lora"""
    voice = core.load_voice(vid)
    sents = split_sentences(text.strip(), params["default_emo"], params["auto_emo"])
    if not sents:
        raise ValueError("请输入要朗读的文字")
    seed = int(params["seed"]) if params.get("seed") is not None and int(params["seed"]) >= 0 else random.randint(0, 2 ** 31 - 1)
    tid = time.strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 999):03d}"
    os.makedirs(_dir(tid))
    name = re.sub(r'[\\/:*?"<>|\s]', "", voice["name"])[:20]
    stem = os.path.join(core.OUTPUTS, f"{name}_{engine}_{tid[:15]}")
    meta = {"id": tid, "vid": vid, "voice_name": voice["name"], "engine": engine,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "params": dict(params, seed=seed),
            "wav": stem + ".wav", "srt": stem + ".srt", "sentences": []}
    t0 = time.time()
    try:
        for i, (emo, s) in enumerate(sents):
            f = f"s{i + 1:03d}.wav"
            log(f"第 {i + 1}/{len(sents)} 句【{emo}】{s[:20]}")
            used, hit = ENG.synth_sentence(engine, voice, emo, s, params, seed + i, os.path.join(_dir(tid), f),
                                           params.get("use_lora", True))
            meta["sentences"].append({"text": s, "emotion": emo, "file": f, "seed": seed + i, "ref_emotion": used, "hit": hit})
    except Exception:
        shutil.rmtree(_dir(tid), ignore_errors=True)
        raise
    meta["sec"] = round(time.time() - t0, 1)
    return build(meta)


def regen(tid, idx, text=None, emotion=None, seed=None):
    """重做第 idx 句（可改文字/情感）；不指定种子时随机换一个。"""
    meta = load(tid)
    s = meta["sentences"][idx]
    if text and text.strip():
        s["text"] = text.strip()
    if emotion:
        s["emotion"] = emotion
    s["seed"] = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2 ** 31 - 1)
    voice = core.load_voice(meta["vid"])
    new = f"s{idx + 1:03d}_{int(time.time()) % 100000}.wav"
    used, hit = ENG.synth_sentence(meta["engine"], voice, s["emotion"], s["text"], meta["params"], s["seed"],
                                   os.path.join(_dir(tid), new), meta["params"].get("use_lora", True))
    try:
        os.remove(os.path.join(_dir(tid), s["file"]))
    except OSError:
        pass
    s.update(file=new, ref_emotion=used, hit=hit)
    return build(meta)


def sentence_path(meta, idx):
    return os.path.join(_dir(meta["id"]), meta["sentences"][idx]["file"])


def table(meta):
    """给界面表格用的行。"""
    rows = []
    for i, s in enumerate(meta["sentences"]):
        emo = s["emotion"] + ("" if s.get("hit", True) else f"（用{s.get('ref_emotion', '平静')}录音）")
        rows.append([i + 1, emo, s["text"], s.get("dur", 0), f"{s.get('start', 0):.1f}s"])
    return rows


def delete(tid):
    meta = load(tid)
    for p in (meta["wav"], meta["srt"]):
        try:
            os.remove(p)
        except OSError:
            pass
    shutil.rmtree(_dir(tid), ignore_errors=True)
