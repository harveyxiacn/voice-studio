# -*- coding: utf-8 -*-
"""声音工坊后端：项目管理、切片、识别、训练、语音包打包。"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
GSV = os.path.join(ROOT, "GPT-SoVITS")
PY = os.path.join(GSV, "runtime", "python.exe")
WORK = os.path.join(ROOT, "workspace")
VOICES = os.path.join(ROOT, "voices")
OUTPUTS = os.path.join(ROOT, "outputs")
for _d in (WORK, VOICES, OUTPUTS):
    os.makedirs(_d, exist_ok=True)

VERSION = "v2ProPlus"
PRE_S2G = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
PRE_S2D = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
PRE_S1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
BERT_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

SR = 32000
MIN_SEG, MAX_SEG = 1.5, 12.0          # 训练切片长度范围（秒）
REF_MIN, REF_MAX = 3.2, 9.8           # GPT-SoVITS 参考音频必须在 3~10 秒之间
REF_LONG_MAX = 15.0                   # VoxCPM2 / CosyVoice3 可用更长的参考
RECOMMEND_MIN_SEC = 60                # 建议最短总时长
GOOD_SEC = 5 * 60                     # 效果较好的总时长

LANGS = {"普通话": "zh", "粤语": "yue", "英语": "en"}
LANG_NAMES = {v: k for k, v in LANGS.items()}
EMOTIONS = ["平静", "开心", "兴奋", "温柔", "悲伤", "生气", "惊讶", "严肃"]
# emotion2vec 标签 -> 中文情感
EMO2VEC_MAP = {
    "neutral": "平静", "happy": "开心", "sad": "悲伤", "angry": "生气",
    "surprised": "惊讶", "fearful": "悲伤", "disgusted": "生气",
}


# ------------------------------------------------------------------ 项目
def _safe_id(name):
    s = re.sub(r"[^A-Za-z0-9]", "", name)[:12]
    return (s or "voice") + time.strftime("_%m%d%H%M%S")


def list_projects():
    out = []
    for pid in sorted(os.listdir(WORK)):
        p = os.path.join(WORK, pid, "project.json")
        if os.path.exists(p):
            out.append((load_project(pid)["name"], pid))
    return out


def project_dir(pid):
    return os.path.join(WORK, pid)


def load_project(pid):
    with open(os.path.join(project_dir(pid), "project.json"), encoding="utf-8") as f:
        return json.load(f)


def save_project(proj):
    d = project_dir(proj["id"])
    tmp = os.path.join(d, "project.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(proj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(d, "project.json"))


def create_project(name):
    name = name.strip()
    if not name:
        raise ValueError("请先填写声音名称")
    for n, pid in list_projects():
        if n == name:
            return load_project(pid)
    pid = _safe_id(name)
    for sub in ("raw", "slices"):
        os.makedirs(os.path.join(project_dir(pid), sub), exist_ok=True)
    proj = {"id": pid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "batches": 0, "segments": []}
    save_project(proj)
    return proj


def project_stats(proj):
    segs = proj["segments"]
    kept = [s for s in segs if s["keep"]]
    total = sum(s["dur"] for s in kept)
    checked = sum(1 for s in kept if s.get("checked"))
    return {"n": len(segs), "kept": len(kept), "sec": total, "checked": checked}


def duration_advice(sec):
    if sec < 10:
        return "❌ 有效人声不足 10 秒，无法使用，请补充录音。"
    if sec < RECOMMEND_MIN_SEC:
        return f"⚠️ 有效人声 {sec:.0f} 秒，低于建议的最短 {RECOMMEND_MIN_SEC} 秒。可以先做「零样本语音包」试听，训练效果会比较一般，建议补充到 1 分钟以上。"
    if sec < GOOD_SEC:
        return f"✅ 有效人声 {sec/60:.1f} 分钟，可以训练。补充到 5 分钟以上会更像本人。"
    return f"✅ 有效人声 {sec/60:.1f} 分钟，素材充足，没有上限，越多越稳定。"


# ------------------------------------------------------------------ 切片
def _load_audio(path):
    sys.path.insert(0, GSV)
    from tools.my_utils import load_audio
    return load_audio(path, SR)


def _rms(y, hop):
    n = len(y) // hop
    if n == 0:
        return np.zeros(1)
    frames = y[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def _split_long(chunk, max_sec=MAX_SEG, min_sec=3.0):
    """超长片段在最安静的位置继续切，保证每段不超过 max_sec。"""
    hop = SR // 100
    out = []
    while len(chunk) > max_sec * SR:
        rms = _rms(chunk, hop)
        lo, hi = int(min_sec * 100), int(max_sec * 100)
        cut = lo + int(np.argmin(rms[lo:hi])) if hi > lo else hi
        out.append(chunk[: cut * hop])
        chunk = chunk[cut * hop:]
    out.append(chunk)
    return out


def slice_audio(path, threshold=-34, min_interval=300):
    sys.path.insert(0, os.path.join(GSV, "tools"))
    from slicer2 import Slicer
    audio = _load_audio(path)
    slicer = Slicer(sr=SR, threshold=threshold, min_length=4000, min_interval=min_interval,
                    hop_size=10, max_sil_kept=500)
    pieces = []
    for chunk, _s, _e in slicer.slice(audio):
        for c in _split_long(np.asarray(chunk, dtype=np.float32)):
            if len(c) >= MIN_SEG * SR:
                pieces.append(c)
    # 响度归一（与 GPT-SoVITS 官方一致：max=0.9, alpha=0.25）
    res = []
    for c in pieces:
        m = np.abs(c).max()
        if m < 1e-4:
            continue
        if m > 1:
            c = c / m
        c = (c / m * (0.9 * 0.25)) + (1 - 0.25) * c
        res.append(c)
    return len(audio) / SR, res


PUNCT = "，。！？、；：,.!?;:…"
SEG_MIN, SEG_MAX, SEG_TARGET = 3.0, 10.0, 7.0   # 按句切分：每段 3~10 秒，尽量靠近 7 秒


def sentence_timestamps(path, lang, log):
    """SenseVoice 逐字时间戳：[[字, 开始, 结束], ...]；不可用或失败时返回 None。"""
    if not sensevoice_ready():
        return None
    out = os.path.join(WORK, f"_ts_{time.strftime('%H%M%S')}_{os.getpid()}.json")
    code = run_cmd([os.path.join(ROOT, "engines", "sensevoice_asr.py"), path, out, "--lang", lang, "--timestamps"],
                   lambda s: log("  [SenseVoice] " + s[-160:]) if ("rror" in s or "Traceback" in s) else None,
                   py=os.path.join(ROOT, "engines", "voxcpm_env", "Scripts", "python.exe"), cwd=os.path.join(ROOT, "engines"))
    if code != 0 or not os.path.exists(out):
        return None
    with open(out, encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    os.remove(out)
    return chars or None


def sentence_segments(chars, total_dur):
    """按句切分：只在标点处（句子/分句边界）下刀，切点取前后两个字之间的中点，组合成 3~10 秒的段。
    返回 [(开始, 结束, 识别文字), ...]"""
    words = [c for c in chars if c[0] not in PUNCT]
    if not words:
        return []
    # 分句：标点处断开，记录每个分句的首尾字时间
    clauses, cur = [], []
    for c in chars:
        if c[0] in PUNCT:
            if cur:
                cur.append(c)
                clauses.append(cur)
                cur = []
        else:
            cur.append(c)
    if cur:
        clauses.append(cur)
    def split_long(cl):
        """没有标点的超长分句：在字与字之间停顿最长处再切。"""
        ws = [c for c in cl if c[0] not in PUNCT]
        if len(ws) < 4 or ws[-1][2] - ws[0][1] <= SEG_MAX:
            return [cl]
        lo, hi = 2, len(ws) - 2
        k = max(range(lo, hi), key=lambda i: ws[i][1] - ws[i - 1][2])
        cut = cl.index(ws[k])
        return split_long(cl[:cut]) + split_long(cl[cut:])

    info = []
    for cl0 in clauses:
        for cl in split_long(cl0):
            ws = [c for c in cl if c[0] not in PUNCT]
            if ws:
                info.append({"b": ws[0][1], "e": ws[-1][2], "text": "".join(c[0] for c in cl)})
    # 贪心组合：累计到 >= 目标长度或再加会超过上限时收尾
    segs, grp = [], []
    for k, c in enumerate(info):
        if grp and (c["e"] - grp[0]["b"] > SEG_MAX or (info[k - 1]["e"] - grp[0]["b"] >= SEG_TARGET)):
            segs.append(grp)
            grp = []
        grp.append(c)
    if grp:
        if segs and grp[-1]["e"] - grp[0]["b"] < SEG_MIN and grp[-1]["e"] - segs[-1][0]["b"] <= SEG_MAX + 2:
            segs[-1] += grp  # 最后一小段并入前一段
        else:
            segs.append(grp)
    out = []
    for i, g in enumerate(segs):
        prev_e = segs[i - 1][-1]["e"] if i else 0.0
        next_b = segs[i + 1][0]["b"] if i + 1 < len(segs) else total_dur
        b = max(0.0, (prev_e + g[0]["b"]) / 2 if i else g[0]["b"] - 0.25)
        e = min(total_dur, (g[-1]["e"] + next_b) / 2 if i + 1 < len(segs) else g[-1]["e"] + 0.35)
        out.append((b, e, "".join(c["text"] for c in g)))
    return out


def align_transcript(segs, transcript):
    """把用户给的准确文稿按识别结果逐字对齐，分配到各段。返回每段的正确文字。"""
    import difflib
    asr_chars, owner = [], []
    for i, (_, _, t) in enumerate(segs):
        for ch in t:
            if ch not in PUNCT:
                asr_chars.append(ch)
                owner.append(i)
    tr = [(k, ch) for k, ch in enumerate(transcript) if ch not in PUNCT and not ch.isspace()]
    sm = difflib.SequenceMatcher(None, "".join(asr_chars), "".join(ch for _, ch in tr), autojunk=False)
    tr_owner = [None] * len(tr)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        for j in range(b0, b1):
            if tag in ("equal", "replace") and a1 > a0:
                tr_owner[j] = owner[min(a0 + (j - b0) * (a1 - a0) // max(b1 - b0, 1), a1 - 1)]
            elif a0 > 0:
                tr_owner[j] = owner[min(a0, len(owner) - 1)]
            else:
                tr_owner[j] = 0
    # 填补缺失，并保证段号单调不减
    last = 0
    for j in range(len(tr_owner)):
        tr_owner[j] = max(last, tr_owner[j] if tr_owner[j] is not None else last)
        last = tr_owner[j]
    # 按原文稿（含标点）切成各段：每段取「首字」到「下一段首字」之间的原文
    starts = {}
    for j, (k, _) in enumerate(tr):
        starts.setdefault(tr_owner[j], k)
    texts, keys = [], sorted(starts)
    for n, i in enumerate(keys):
        end = starts[keys[n + 1]] if n + 1 < len(keys) else len(transcript)
        texts.append((i, transcript[starts[i]:end].strip()))
    res = [""] * len(segs)
    for i, t in texts:
        res[i] = t
    return res


def _norm_chunk(c):
    """与 GPT-SoVITS 官方一致的响度归一（max=0.9, alpha=0.25）。"""
    m = float(np.abs(c).max()) if len(c) else 0
    if m < 1e-4:
        return None
    if m > 1:
        c = c / m
        m = 1.0
    return (c / m * (0.9 * 0.25)) + (1 - 0.25) * c


def add_audio(proj, files, lang_name, log, transcript=None):
    """上传音频 -> 切片 -> 语音识别 -> 情感识别，追加到项目。
    切片优先「按句切分」（SenseVoice 逐字时间戳，只在句子/分句边界下刀，不会切到半个字），失败时退回按静音切分。
    transcript：只有一个文件时可提供准确文稿，会逐字对齐分配到各段（这些段直接算已校对）。"""
    from scipy.io import wavfile
    lang = LANGS[lang_name]
    proj["batches"] += 1
    b = proj["batches"]
    d = project_dir(proj["id"])
    new_dir = os.path.join(d, f"_new{b}")
    shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(new_dir)
    total_in = 0.0
    idx = 0
    given = {}
    for f in files:
        src = f if isinstance(f, str) else f.name
        base = os.path.basename(src)
        raw = os.path.join(d, "raw", f"b{b:03d}_{base}")
        shutil.copy(src, raw)
        log(f"切片中：{base}")
        audio = _load_audio(raw)
        dur = len(audio) / SR
        total_in += dur
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            log(f"  ⚠️ 原音频有削波/爆音（峰值 {peak:.2f}），已整体降低音量避免再次失真；有条件请用未爆音的原始录音")
            audio = audio * (0.97 / peak)
        chars = sentence_timestamps(raw, lang, log)
        segs = sentence_segments(chars, dur) if chars else []
        if segs:
            texts = align_transcript(segs, transcript) if (transcript and len(files) == 1) else [None] * len(segs)
            n0 = idx
            for (sb, se, _), t in zip(segs, texts):
                c = _norm_chunk(audio[int(sb * SR):int(se * SR)].astype(np.float32))
                if c is None or len(c) < MIN_SEG * SR:
                    continue
                idx += 1
                name = f"b{b:03d}_{idx:05d}.wav"
                wavfile.write(os.path.join(new_dir, name), SR, (c * 32767).astype(np.int16))
                if t:
                    given[name] = t
            log(f"  {base}：{dur / 60:.1f} 分钟 → 按句切成 {idx - n0} 段" + ("（已按你提供的文稿对齐）" if given else ""))
        else:
            _, chunks = slice_audio(raw)
            for c in chunks:
                idx += 1
                wavfile.write(os.path.join(new_dir, f"b{b:03d}_{idx:05d}.wav"), SR, (c * 32767).astype(np.int16))
            log(f"  {base}：{dur / 60:.1f} 分钟 → 按静音切成 {len(chunks)} 段")
    if idx == 0:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise RuntimeError("没有切出有效人声，请检查音频是否有声音或过于安静。")

    log("语音识别中（首次运行需加载模型，稍等）…")
    asr = run_asr_dual(new_dir, lang, log)
    if not asr:
        log("⚠️ 自动识别失败或无结果，请在校对页手动填写文稿。")
    log("情感识别中…")
    emos = run_emotion(new_dir, log)

    for name in sorted(os.listdir(new_dir)):
        if not name.endswith(".wav"):
            continue
        dst = os.path.join(d, "slices", name)
        shutil.move(os.path.join(new_dir, name), dst)
        sr, y = wavfile.read(dst)
        a = asr.get(name, {})
        text = given.get(name) or a.get("text", "")
        proj["segments"].append({
            "file": name, "dur": round(len(y) / sr, 2), "lang": lang,
            "text": text, "asr": a.get("text", ""), "alt": a.get("alt", ""),
            "conf": 1.0 if name in given else a.get("conf", 0.0),
            "emotion": emos.get(name) or a.get("sv_emotion") or "平静",
            "keep": bool(text.strip()), "checked": name in given,
        })
    shutil.rmtree(new_dir, ignore_errors=True)
    save_project(proj)
    log("更新声纹（用于以后自动认人）…")
    update_voiceprint(proj["id"], log)
    return total_in, idx


def _env(extra=None):
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "version": VERSION,
                "is_half": "True", "_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz",
                "no_proxy": "localhost, 127.0.0.1, ::1", "all_proxy": ""})
    env["PATH"] = GSV + os.pathsep + os.path.join(GSV, "runtime") + os.pathsep + env.get("PATH", "")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_cmd(args, log, env_extra=None, py=None, cwd=None):
    """运行子进程并逐行输出日志；返回退出码。默认用 GPT-SoVITS 自带的 Python。"""
    cmd = [py] + args if py else [PY, "-s"] + args
    p = subprocess.Popen(cmd, cwd=cwd or GSV, env=_env(env_extra), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    RUNNING.append(p)
    try:
        for raw in iter(p.stdout.readline, b""):
            # 进度条（tqdm）用 \r 原地刷新，会和下一行输出粘在一起，按 \r 拆开逐段处理
            for part in raw.decode("utf-8", "replace").split("\r"):
                part = part.rstrip()
                if part:
                    log(part)
        p.wait()
    finally:
        RUNNING.remove(p)
    return p.returncode


RUNNING = []


def stop_all():
    for p in list(RUNNING):
        subprocess.call(f"taskkill /t /f /pid {p.pid}", shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


SV_MODEL = os.path.join(ROOT, "engines", "models", "SenseVoiceSmall", "model.pt")
_NON_TEXT = re.compile(r"[^一-鿿㐀-䶿A-Za-z0-9，。！？、；：“”‘’（）《》…,.!?;:'\"()\s-]")


def sensevoice_ready():
    return os.path.exists(os.path.join(ROOT, "engines", "voxcpm_env", "Scripts", "python.exe")) and os.path.exists(SV_MODEL)


def run_sensevoice(wav_dir, lang, log, device="cuda:0"):
    """SenseVoice 识别（带标点），返回 {文件名: {"text", "emotion", "lang"}}；不可用时返回空。"""
    if not sensevoice_ready():
        return {}
    out = wav_dir.rstrip("\\/") + "_sv.json"
    code = run_cmd([os.path.join(ROOT, "engines", "sensevoice_asr.py"), wav_dir, out, "--lang", lang, "--device", device],
                   lambda s: log("  [SenseVoice] " + s[-160:]) if ("rror" in s or "Traceback" in s) else None,
                   py=os.path.join(ROOT, "engines", "voxcpm_env", "Scripts", "python.exe"), cwd=os.path.join(ROOT, "engines"))
    res = {}
    if code == 0 and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        os.remove(out)
        for v in res.values():
            v["text"] = _NON_TEXT.sub("", v["text"]).strip()
    return res


def run_sensevoice_raw(wav_dir, lang, log, device="cuda:0"):
    """评分用：不做数字转写的识别结果 {文件名: 文字}（「两千零二十六」保持汉字，才能和「二零二六」区分）。"""
    if not sensevoice_ready():
        return {}
    out = wav_dir.rstrip("\\/") + f"_raw_{lang}.json"
    code = run_cmd([os.path.join(ROOT, "engines", "sensevoice_asr.py"), wav_dir, out, "--lang", lang, "--device", device,
                    "--no-itn"], lambda s: log("  [SenseVoice] " + s[-160:]) if ("rror" in s or "Traceback" in s) else None,
                   py=os.path.join(ROOT, "engines", "voxcpm_env", "Scripts", "python.exe"), cwd=os.path.join(ROOT, "engines"))
    res = {}
    if code == 0 and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            res = {k: _NON_TEXT.sub("", v["text"]).strip() for k, v in json.load(f).items()}
        os.remove(out)
    return res


def _norm(t):
    return re.sub(r"[^一-鿿A-Za-z0-9]", "", t or "").lower()


def asr_confidence(a, b):
    """两个识别结果的一致度（0~1），作为文稿可信度。"""
    import difflib
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 2)


def run_asr_dual(wav_dir, lang, log):
    """双识别：粤语以 SenseVoice 为主（带标点），普通话以 Paraformer 为主；另一个作为第二意见算可信度。
    返回 {文件名: {"text", "alt", "conf", "sv_emotion"}}"""
    log("  SenseVoice 识别中…")
    sv = run_sensevoice(wav_dir, lang if lang in ("zh", "yue", "en") else "auto", log)
    log("  FunASR 识别中…")
    fa = run_asr(wav_dir, lang, log) if lang in ("zh", "yue") else {}
    res = {}
    for name in set(sv) | set(fa):
        s_txt = sv.get(name, {}).get("text", "")
        f_txt = fa.get(name, "")
        main, alt = (s_txt or f_txt, f_txt) if lang in ("yue", "en") else (f_txt or s_txt, s_txt)
        res[name] = {"text": main, "alt": alt if alt != main else "",
                     "conf": asr_confidence(main, alt) if (main and alt) else (0.5 if main else 0.0),
                     "sv_emotion": sv.get(name, {}).get("emotion")}
    return res


def run_asr(wav_dir, lang, log):
    out_dir = wav_dir + "_asr"
    shutil.rmtree(out_dir, ignore_errors=True)
    if lang in ("zh", "yue"):
        args = ["tools/asr/funasr_asr.py", "-i", wav_dir, "-o", out_dir, "-s", "large", "-l", lang, "-p", "float32"]
    else:
        args = ["tools/asr/fasterwhisper_asr.py", "-i", wav_dir, "-o", out_dir, "-s", "large-v3", "-l", lang, "-p", "float16"]
    code = run_cmd(args, lambda s: log("  [ASR] " + s[-160:]) if ("加载" in s or "Error" in s or "rror" in s) else None)
    texts = {}
    if os.path.isdir(out_dir):
        for fn in os.listdir(out_dir):
            if fn.endswith(".list"):
                with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("|")
                        if len(parts) >= 4:
                            texts[os.path.basename(parts[0])] = "|".join(parts[3:]).strip()
    shutil.rmtree(out_dir, ignore_errors=True)
    return texts


def run_emotion(wav_dir, log):
    out = os.path.join(wav_dir, "_emotion.json")
    code = run_cmd([os.path.join(APP_DIR, "emotion_tag.py"), wav_dir, out],
                   lambda s: log("  [情感] " + s[-160:]) if s.startswith("[") else None)
    emos = {}
    if code == 0 and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            emos = json.load(f)
        os.remove(out)
    else:
        log("⚠️ 情感自动识别不可用，已全部标记为「平静」，可在校对页手动修改。")
    return emos


# ------------------------------------------------------------------ 人声分离
def separate_files(paths, log, dereverb=False):
    """BS-Roformer 去背景音乐/噪音（可选去混响）。返回与输入一一对应的人声 wav 路径。"""
    out_dir = os.path.join(WORK, "_separated", time.strftime("%Y%m%d_%H%M%S"))
    outs = []

    def f(s):
        if s.startswith("@@OUT@@"):
            outs.append(s[7:])
        elif s.startswith("[sep]"):
            log(s[6:])
        elif "rror" in s or "Traceback" in s:
            log("  " + s[-200:])

    code = run_cmd([os.path.join(APP_DIR, "separate.py"), out_dir] + list(paths) + (["--dereverb"] if dereverb else []), f)
    if code != 0 or len(outs) != len(paths):
        raise RuntimeError("人声分离失败，请查看日志。")
    return outs


# ------------------------------------------------------------------ 说话人分离
def diarize_files(paths, log, threshold=0.55, num=0):
    out = os.path.join(WORK, f"_diar_{time.strftime('%Y%m%d_%H%M%S')}.json")
    args = [os.path.join(APP_DIR, "diarize.py"), out] + list(paths) + ["--threshold", str(threshold)]
    if num:
        args += ["--num", str(int(num))]
    code = run_cmd(args, lambda s: log(s[7:]) if s.startswith("[diar]") else (log("  " + s[-200:]) if "rror" in s else None))
    if code != 0 or not os.path.exists(out):
        raise RuntimeError("说话人识别失败，请查看日志。")
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    os.remove(out)
    return res


_audio_cache = {}


def _file_audio(path):
    if path not in _audio_cache:
        _audio_cache.clear()  # 只缓存最近一个文件，避免长视频占满内存
        _audio_cache[path] = _load_audio(path)
    return _audio_cache[path]


def speaker_audio(res, spk_ids, out_path, max_sec=None, gap=0.6):
    """把指定说话人的所有发言段拼成一个 wav（段间插入静音，便于后续按静音切片）。"""
    from scipy.io import wavfile
    turns = [t for t in res["turns"] if t[3] in spk_ids]
    if max_sec:  # 试听：优先挑较长的段
        turns = sorted(turns, key=lambda t: -(t[2] - t[1]))
        picked, tot = [], 0.0
        for t in turns:
            if tot >= max_sec:
                break
            picked.append(t)
            tot += t[2] - t[1]
        turns = sorted(picked, key=lambda t: (t[0], t[1]))
    pieces, sil = [], np.zeros(int(SR * gap), np.float32)
    for fi, b, e, _ in sorted(turns, key=lambda t: (t[0], t[1])):
        # 有人声分离结果时从干净人声里截取（时间轴与原音频一致）
        wav = _file_audio(res["files"][fi].get("audio") or res["files"][fi]["path"])
        pieces += [wav[int(b * SR):int(e * SR)], sil]
    audio = np.concatenate(pieces) if pieces else np.zeros(SR, np.float32)
    wavfile.write(out_path, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return len(audio) / SR


VP_MATCH = 0.5  # 说话人与已有声音的声纹相似度 ≥ 此值即建议为同一人


def update_voiceprint(pid, log=lambda m: None):
    proj = load_project(pid)
    files = [os.path.join(project_dir(pid), "slices", s["file"]) for s in proj["segments"] if s["keep"]]
    if not files:
        return
    run_cmd([os.path.join(APP_DIR, "diarize.py"), "voiceprint", os.path.join(project_dir(pid), "voiceprint.npy")] + files[:80],
            lambda s: log(s[7:]) if s.startswith("[diar]") else None)


def match_voiceprints(res):
    """为每位说话人找最像的已有声音：[(项目名, 相似度) 或 None]"""
    vps = []
    for name, pid in list_projects():
        p = os.path.join(project_dir(pid), "voiceprint.npy")
        if os.path.exists(p):
            vps.append((name, np.load(p)))
    out = []
    for c in res.get("centroids") or []:
        c = np.asarray(c)
        best = max(((n, float(v @ c)) for n, v in vps), key=lambda x: x[1], default=None)
        out.append(best if best and best[1] >= VP_MATCH else None)
    return out


def speaker_hints(res):
    """每位说话人的提示：时长是否够用、和谁声纹接近。"""
    S = np.array(res.get("similarity") or [[1.0]])
    hints = []
    for sp in res["speakers"]:
        i, h = sp["id"], []
        if sp["dur"] < 10:
            h.append(f"⚠️ 只有 {sp['dur']:.0f} 秒，不够做语音包（建议 ≥1 分钟），可取消勾选或合并")
        elif sp["dur"] < RECOMMEND_MIN_SEC:
            h.append(f"时长 {sp['dur']:.0f} 秒，可以先做零样本语音包，建议补充到 1 分钟以上")
        for j in range(len(res["speakers"])):
            if j != i and S[i, j] >= 0.4:
                h.append(f"🔍 与说话人{j + 1} 声纹较接近（{S[i, j]:.2f}），如果是同一个人，请填相同的名字")
        hints.append("　".join(h))
    return hints


def process_speakers(res, assign, log):
    """assign: [{"spk": id, "name": 名字, "lang": 语言名, "keep": bool}]；同名的说话人合并成一个声音。"""
    groups = {}
    for a in assign:
        if a["keep"] and a["name"].strip():
            g = groups.setdefault(a["name"].strip(), {"spk": [], "lang": a["lang"]})
            g["spk"].append(a["spk"])
    if not groups:
        raise ValueError("没有选择任何说话人")
    base = os.path.splitext(os.path.basename(res["files"][0]["path"]))[0][:30]
    made = []
    for name, g in groups.items():
        proj = create_project(name)
        log(f"━━ {name}（说话人 {'、'.join(str(s + 1) for s in g['spk'])}）")
        tmp_dir = os.path.join(project_dir(proj["id"]), "_speaker")
        os.makedirs(tmp_dir, exist_ok=True)
        wav = os.path.join(tmp_dir, f"{base}_{'_'.join(str(s + 1) for s in g['spk'])}.wav")
        sec = speaker_audio(res, set(g["spk"]), wav)
        log(f"  提取该说话人的语音 {sec / 60:.1f} 分钟")
        add_audio(proj, [wav], g["lang"], log)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        proj = load_project(proj["id"])
        try:
            meta = build_voice(proj)
            log(f"  ✅ 已生成语音包（参考录音 {len(meta['refs'])} 条）")
        except RuntimeError as e:
            log(f"  ⚠️ 暂未生成语音包：{e}")
        made.append(proj["id"])
    return made


# ------------------------------------------------------------------ 训练
def _latest(folder, pattern):
    best, best_ep = None, -1
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            m = re.search(pattern, fn)
            if m and int(m.group(1)) > best_ep:
                best, best_ep = os.path.join(folder, fn), int(m.group(1))
    return best


def train(proj, sovits_epochs, gpt_epochs, batch_size, log, progress=None):
    progress = progress or (lambda pct, text: None)
    import yaml
    pid = proj["id"]
    d = project_dir(pid)
    segs = [s for s in proj["segments"] if s["keep"] and s["text"].strip()]
    if not segs:
        raise RuntimeError("没有可用于训练的片段（需要保留且有文稿）。")
    exp = os.path.join(d, "train").replace("\\", "/")
    shutil.rmtree(exp, ignore_errors=True)
    os.makedirs(f"{exp}/logs_s2_{VERSION}")
    os.makedirs(f"{exp}/logs_s1_{VERSION}")
    sov_dir = os.path.join(d, "weights", "sovits").replace("\\", "/")
    gpt_dir = os.path.join(d, "weights", "gpt").replace("\\", "/")
    for x in (sov_dir, gpt_dir):
        shutil.rmtree(x, ignore_errors=True)
        os.makedirs(x)
    lst = os.path.join(exp, "train.list").replace("\\", "/")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            text = s["text"].replace("|", " ").replace("\n", " ").strip()
            f.write(f"{os.path.join(d, 'slices', s['file'])}|{pid}|{s['lang'].upper()}|{text}\n")
    log(f"训练集：{len(segs)} 段，共 {sum(s['dur'] for s in segs)/60:.1f} 分钟")

    base = {"inp_text": lst, "inp_wav_dir": "", "exp_name": pid, "opt_dir": exp,
            "i_part": 0, "all_parts": 1, "is_half": "True"}
    steps = [
        ("1/6 文本与 BERT 特征", "GPT_SoVITS/prepare_datasets/1-get-text.py", {"bert_pretrained_dir": BERT_DIR}),
        ("2/6 SSL 声学特征", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", {"cnhubert_base_dir": HUBERT_DIR, "sv_path": SV_PATH}),
        ("3/6 说话人特征", "GPT_SoVITS/prepare_datasets/2-get-sv.py", {"cnhubert_base_dir": HUBERT_DIR, "sv_path": SV_PATH}),
        ("4/6 语义 token", "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
         {"pretrained_s2G": PRE_S2G, "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json"}),
    ]
    for k, (title, script, extra) in enumerate(steps):
        log("▶ " + title)
        progress(0.025 * k, "数据预处理：" + title[4:])
        env = dict(base, **extra)
        if run_cmd([script], lambda s: log("  " + s[-200:]) if ("rror" in s or "Traceback" in s) else None, env) != 0:
            raise RuntimeError(f"{title} 失败，请查看日志。")
    # 合并分片输出（单卡只有 part0）
    os.replace(f"{exp}/2-name2text-0.txt", f"{exp}/2-name2text.txt")
    with open(f"{exp}/6-name2semantic-0.tsv", encoding="utf-8") as f:
        sem = f.read().strip("\n")
    with open(f"{exp}/6-name2semantic.tsv", "w", encoding="utf-8") as f:
        f.write("item_name\tsemantic_audio\n" + sem + "\n")
    os.remove(f"{exp}/6-name2semantic-0.tsv")

    # SoVITS（音色）
    log(f"▶ 5/6 训练 SoVITS 音色模型（{sovits_epochs} 轮）")
    with open(os.path.join(GSV, f"GPT_SoVITS/configs/s2{VERSION}.json"), encoding="utf-8") as f:
        c = json.load(f)
    c["train"].update({"batch_size": batch_size, "epochs": sovits_epochs, "text_low_lr_rate": 0.4,
                       "pretrained_s2G": PRE_S2G, "pretrained_s2D": PRE_S2D, "if_save_latest": True,
                       "if_save_every_weights": True, "save_every_epoch": sovits_epochs,
                       "gpu_numbers": "0", "grad_ckpt": False, "lora_rank": 32})
    c["model"]["version"] = VERSION
    c["data"]["exp_dir"] = c["s2_ckpt_dir"] = exp
    c.update({"save_weight_dir": sov_dir, "name": pid, "version": VERSION})
    cfg = os.path.join(exp, "tmp_s2.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(c, f)
    code = run_cmd(["GPT_SoVITS/s2_train.py", "--config", cfg],
                   _train_log(log, progress, r"====> Epoch: (\d+)", sovits_epochs, 0.10, 0.50, "音色模型"))
    sovits = _latest(sov_dir, r"_e(\d+)_s\d+\.pth$")
    if code != 0 or not sovits:
        raise RuntimeError("SoVITS 训练失败。若提示显存不足，请关闭剪映等占显存的软件或调小批量。")

    # GPT（语气、节奏、情感）
    log(f"▶ 6/6 训练 GPT 语气模型（{gpt_epochs} 轮）")
    with open(os.path.join(GSV, "GPT_SoVITS/configs/s1longer-v2.yaml"), encoding="utf-8") as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    y["train"].update({"batch_size": batch_size, "epochs": gpt_epochs, "save_every_n_epoch": gpt_epochs,
                       "if_save_every_weights": True, "if_save_latest": True, "if_dpo": False,
                       "half_weights_save_dir": gpt_dir, "exp_name": pid})
    y["pretrained_s1"] = PRE_S1
    y["train_semantic_path"] = f"{exp}/6-name2semantic.tsv"
    y["train_phoneme_path"] = f"{exp}/2-name2text.txt"
    y["output_dir"] = f"{exp}/logs_s1_{VERSION}"
    cfg = os.path.join(exp, "tmp_s1.yaml")
    with open(cfg, "w", encoding="utf-8") as f:
        yaml.dump(y, f, default_flow_style=False)
    code = run_cmd(["GPT_SoVITS/s1_train.py", "--config_file", cfg],
                   _train_log(log, progress, r"Epoch (\d+):", gpt_epochs, 0.60, 0.40, "语气模型", zero_based=True))
    gpt = _latest(gpt_dir, r"-e(\d+)\.ckpt$")
    if code != 0 or not gpt:
        raise RuntimeError("GPT 训练失败，请查看日志。")
    return sovits, gpt


ENG_DIR = os.path.join(ROOT, "engines")
VOX_PY = os.path.join(ENG_DIR, "voxcpm_env", "Scripts", "python.exe")


def voxcpm_ready():
    return os.path.exists(VOX_PY) and os.path.exists(os.path.join(ENG_DIR, "models", "VoxCPM2", "model.safetensors"))


def suggest_voxcpm_iters(sec):
    """按素材时长推荐步数：素材少时少练，避免过拟合。"""
    return int(min(1500, max(400, 400 + sec / 60 * 60)) // 50 * 50)


def train_voxcpm(proj, iters, log, progress=None):
    if not voxcpm_ready():
        raise RuntimeError("VoxCPM2 引擎未安装")
    log(f"▶ VoxCPM2 LoRA 微调：{iters} 步（每步 8 条样本）")
    lora, last = [None], [0.0]
    progress = progress or (lambda pct, text: None)
    progress(0.01, "加载模型与数据…")

    def f(s):
        if s.startswith("@@LORA@@"):
            lora[0] = s[len("@@LORA@@"):]
        elif s.startswith("[train]"):
            m = re.search(r"step (\d+)", s)
            if m:
                progress(0.03 + 0.95 * (int(m.group(1)) + 1) / iters, f"训练第 {int(m.group(1)) + 1}/{iters} 步")
            if time.time() - last[0] > 20:
                last[0] = time.time()
                log("  " + s.split(", epoch")[0].replace("[train] ", ""))
        elif any(k in s for k in ("训练集", "低显存", "rror", "Traceback", "out of memory")):
            log("  " + s[-200:])

    code = run_cmd([os.path.join(ENG_DIR, "voxcpm_lora_train.py"), proj["id"], "--iters", str(int(iters))], f,
                   py=VOX_PY, cwd=ENG_DIR)
    if code != 0 or not lora[0]:
        raise RuntimeError("VoxCPM2 微调失败。若提示 out of memory，请关闭所有占显存的软件后重试。")
    return lora[0]


def _train_log(log, progress=None, pat=None, total=1, base=0.0, span=0.0, stage="", zero_based=False):
    """过滤训练日志；按 pat 解析当前轮次并换算成总进度 base + span × 轮次/总轮数。"""
    last = [0.0]

    def f(s):
        if progress and pat:
            m = re.search(pat, s)
            if m:
                ep = int(m.group(1)) + (1 if zero_based else 0)
                progress(base + span * min(ep, total) / max(total, 1), f"{stage}：第 {min(ep, total)}/{total} 轮")
        # 训练日志很长，只保留关键行，进度条每 5 秒输出一次
        key = ("Epoch" in s or "epoch" in s or "saving" in s or "rror" in s or "Traceback" in s
               or "out of memory" in s)
        if key and ("it/s" not in s and "s/it" not in s or time.time() - last[0] > 5):
            last[0] = time.time()
            log("  " + s[-200:])
    return f


# ------------------------------------------------------------------ 语音包
def list_voices():
    out = []
    for vid in sorted(os.listdir(VOICES)):
        p = os.path.join(VOICES, vid, "voice.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                v = json.load(f)
            tags = [t for t, ok in (("VoxCPM2微调", v.get("voxcpm_lora")), ("SoVITS已训练", v["trained"])) if ok]
            out.append((f"{v['name']}（{'、'.join(tags) or '零样本'}）", vid))
    return out


def load_voice(vid):
    with open(os.path.join(VOICES, vid, "voice.json"), encoding="utf-8") as f:
        v = json.load(f)
    v["dir"] = os.path.join(VOICES, vid)
    # 语音包可能被导入/移动过：LoRA 路径以语音包目录为准
    lora = os.path.join(v["dir"], "voxcpm_lora")
    v["voxcpm_lora"] = lora if os.path.isdir(lora) else None
    return v


# ------------------------------------------------------------------ 管理
def _dir_size(p):
    tot = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return tot


def overview():
    """管理页表格：每个声音项目 / 语音包的状态。"""
    rows, seen = [], set()
    for name, pid in list_projects():
        p = load_project(pid)
        s = project_stats(p)
        vj = os.path.join(VOICES, pid, "voice.json")
        v = load_voice(pid) if os.path.exists(vj) else None
        rows.append([name, s["kept"], f"{s['checked']}/{s['kept']}", round(s["sec"] / 60, 1),
                     "✅" if v else "—", "✅" if v and v.get("voxcpm_lora") else "—", "✅" if v and v["trained"] else "—",
                     sum(1 for x in p["segments"] if x.get("source") == "录音向导"),
                     round((_dir_size(project_dir(pid)) + (_dir_size(os.path.join(VOICES, pid)) if v else 0)) / 2 ** 20),
                     p["created"]])
        seen.add(pid)
    for label, vid in list_voices():
        if vid not in seen:  # 导入的语音包（没有素材项目）
            v = load_voice(vid)
            rows.append([v["name"] + "（导入）", 0, "—", 0, "✅", "✅" if v.get("voxcpm_lora") else "—",
                         "✅" if v["trained"] else "—", 0, round(_dir_size(v["dir"]) / 2 ** 20), v.get("created", "")])
    return rows


def rename_project(pid, new):
    new = new.strip()
    if not new:
        raise ValueError("名字不能为空")
    if any(n == new for n, p in list_projects() if p != pid):
        raise ValueError(f"已有名为「{new}」的声音，可以用「合并」")
    p = load_project(pid)
    p["name"] = new
    save_project(p)
    vj = os.path.join(VOICES, pid, "voice.json")
    if os.path.exists(vj):
        with open(vj, encoding="utf-8") as f:
            v = json.load(f)
        v["name"] = new
        with open(vj, "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False, indent=1)


def delete_project(pid):
    shutil.rmtree(project_dir(pid), ignore_errors=True)
    shutil.rmtree(os.path.join(VOICES, pid), ignore_errors=True)


def move_segments(src, idxs, dst):
    """把 src 项目的若干切片移到 dst 项目（例如识别时分错了人）。"""
    if src == dst:
        return 0
    ps, pd = load_project(src), load_project(dst)
    moved = []
    for i in sorted(set(idxs), reverse=True):
        s = ps["segments"].pop(i)
        name = s["file"]
        while os.path.exists(os.path.join(project_dir(dst), "slices", name)):
            name = "m_" + name
        shutil.move(os.path.join(project_dir(src), "slices", s["file"]), os.path.join(project_dir(dst), "slices", name))
        rec = ps.get("recorded", {})
        for k, v in list(rec.items()):
            if v == s["file"]:
                del rec[k]
        s["file"] = name
        moved.append(s)
    pd["segments"] += moved[::-1]
    save_project(ps)
    save_project(pd)
    return len(moved)


def merge_projects(src, dst):
    """把 src 的全部素材并入 dst，然后删除 src。"""
    n = move_segments(src, range(len(load_project(src)["segments"])), dst)
    delete_project(src)
    return n


def export_voice(vid):
    v = load_voice(vid)
    out_dir = os.path.join(OUTPUTS, "_exports")
    os.makedirs(out_dir, exist_ok=True)
    name = re.sub(r'[\\/:*?"<>|\s]', "", v["name"])
    base = os.path.join(out_dir, f"{name}_语音包_{time.strftime('%Y%m%d')}")
    return shutil.make_archive(base, "zip", v["dir"])


def import_voice(zip_path):
    import zipfile
    tmp = os.path.join(WORK, "_import")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    vj = os.path.join(tmp, "voice.json")
    if not os.path.exists(vj):
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("不是有效的语音包（缺少 voice.json）")
    with open(vj, encoding="utf-8") as f:
        v = json.load(f)
    vid = v["id"]
    if os.path.exists(os.path.join(VOICES, vid)):
        vid = _safe_id(v["name"])
        v["id"] = vid
        with open(vj, "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False, indent=1)
    shutil.move(tmp, os.path.join(VOICES, vid))
    return v["name"]


UV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "uv", "cache")


def disk_items():
    """可清理的项目：[(key, 名称, 说明, 路径列表)]"""
    train_dirs = []
    for _, pid in list_projects():
        for sub in ("train", "weights", "voxcpm_lora", "_speaker"):
            p = os.path.join(project_dir(pid), sub)
            if os.path.isdir(p):
                train_dirs.append(p)
    tmp = [os.path.join(WORK, d) for d in ("_separated", "_preview", "_eval", "_import") if os.path.isdir(os.path.join(WORK, d))]
    return [
        ("downloads", "GPT-SoVITS 安装包", "已解压安装完成，可以删除", [os.path.join(ROOT, "downloads")]),
        ("uv", "Python 下载缓存（C 盘）", "新引擎环境已装好，删除不影响使用", [UV_CACHE]),
        ("train", "训练中间文件", "训练成果已复制进语音包，删除不影响使用；之后重新训练会重新生成", train_dirs),
        ("tmp", "临时文件", "人声分离结果、试听片段等", tmp),
    ]


def disk_usage():
    return [(k, name, desc, sum(_dir_size(p) for p in paths if os.path.exists(p)) / 2 ** 30)
            for k, name, desc, paths in disk_items()]


def cleanup(key):
    for k, _, _, paths in disk_items():
        if k == key:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)


def pick_refs(proj, per_emotion=6):
    """每种情感挑选若干条 3~10 秒、已保留且有文稿的片段作为参考音频，优先 5~8 秒、已校对的。"""
    # VoxCPM2 / CosyVoice3 可用最长 15 秒的参考（信息更多）；GPT-SoVITS 只能用 3~10 秒（合成时再筛）
    cands = [s for s in proj["segments"] if s["keep"] and s["text"].strip() and REF_MIN <= s["dur"] <= REF_LONG_MAX]
    by = {}
    for s in cands:
        by.setdefault(s["emotion"], []).append(s)
    refs = []
    for emo, lst in by.items():
        # 优先：已校对 → 识别可信度高 → 时长接近 7 秒
        lst.sort(key=lambda s: (not s.get("checked"), -round(s.get("conf", 0.5), 1), abs(s["dur"] - 7)))
        # 各语言都要有参考
        for lang in {s["lang"] for s in lst}:
            refs += [s for s in lst if s["lang"] == lang][:per_emotion]
    return refs


def build_voice(proj, sovits=None, gpt=None, voxcpm_lora=None):
    """生成/更新语音包。参考音频按最新校对结果重新挑选；未传入的模型沿用语音包里已有的。"""
    refs = pick_refs(proj)
    if not refs:
        raise RuntimeError("没有 3~10 秒且有文稿的片段可作为参考音频，请补充录音或检查校对。")
    vid = proj["id"]
    vdir = os.path.join(VOICES, vid)
    old = {}
    if os.path.exists(os.path.join(vdir, "voice.json")):
        with open(os.path.join(vdir, "voice.json"), encoding="utf-8") as f:
            old = json.load(f)
    shutil.rmtree(os.path.join(vdir, "refs"), ignore_errors=True)
    os.makedirs(os.path.join(vdir, "refs"))
    if sovits and gpt:
        shutil.copy(sovits, os.path.join(vdir, "sovits.pth"))
        shutil.copy(gpt, os.path.join(vdir, "gpt.ckpt"))
    trained = os.path.exists(os.path.join(vdir, "sovits.pth")) and os.path.exists(os.path.join(vdir, "gpt.ckpt"))
    lora_dir = os.path.join(vdir, "voxcpm_lora")
    if voxcpm_lora:
        shutil.rmtree(lora_dir, ignore_errors=True)
        shutil.copytree(voxcpm_lora, lora_dir)
    # 用户锁定过的参考录音：只要素材还在，就一定保留在参考录音里
    pinned = old.get("pinned") or {}
    have = {s["file"] for s in refs}
    by_file = {s["file"]: s for s in proj["segments"] if s["keep"] and s["text"].strip()}
    for f in pinned.values():
        if f not in have and f in by_file:
            refs.append(by_file[f])
            have.add(f)
    pinned = {k: f for k, f in pinned.items() if f in have}
    for s in refs:
        shutil.copy(os.path.join(project_dir(vid), "slices", s["file"]), os.path.join(vdir, "refs", s["file"]))
    meta = {
        "id": vid, "name": proj["name"], "trained": trained, "version": VERSION,
        "voxcpm_lora": lora_dir if os.path.isdir(lora_dir) else None,
        "created": old.get("created", time.strftime("%Y-%m-%d %H:%M")), "updated": time.strftime("%Y-%m-%d %H:%M"),
        "langs": sorted({s["lang"] for s in proj["segments"] if s["keep"]}),
        "refs": [{"file": s["file"], "text": s["text"], "lang": s["lang"], "emotion": s["emotion"], "dur": s["dur"]}
                 for s in refs],
        "pinned": pinned,
    }
    with open(os.path.join(vdir, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def _update_voice_json(vid, fn):
    p = os.path.join(VOICES, vid, "voice.json")
    with open(p, encoding="utf-8") as f:
        v = json.load(f)
    fn(v)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=1)
    return v


def pin_ref(vid, lang, emotion, ref_file):
    """锁定：这个语音包以后生成「lang + emotion」的句子都用 ref_file 作参考录音。"""
    _update_voice_json(vid, lambda v: v.setdefault("pinned", {}).__setitem__(f"{lang}|{emotion}", ref_file))


def unpin_all(vid):
    _update_voice_json(vid, lambda v: v.__setitem__("pinned", {}))


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).decode().strip().split("\n")[0]
        name, total, used = [x.strip() for x in out.split(",")]
        return name, int(total), int(used)
    except Exception:
        return "未知显卡", 0, 0
