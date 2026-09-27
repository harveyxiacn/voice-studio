# -*- coding: utf-8 -*-
"""自动评分（给用户做参考，不替代人耳）。

一、为什么不是一个「像不像」分数
    实测声纹相似度与人耳评分几乎不相关（秩相关 -0.03~0.47）。人判断「像」时还看语调、节奏、口音。
    所以这里拆成多个**可测量**的维度，每个维度都和「这个人自己的素材」比，并给出具体原因（flags）。

二、维度
    准确度  识别回文字（不做数字转写，才能发现「2026」读成「两千零二十六」）与原文的字错误率
    语速    每秒字数 / 本人素材的每秒字数
    停顿    句中 ≥0.4 秒的停顿数 - 标点数（多出来的算「停顿太多」）
    音高    平均音高与本人的半音差
    起伏    音高起伏（半音标准差）/ 本人的起伏（太平 = 机械）
    音色    声纹相似度（参考，权重低）
    口音    仅粤语：粤语识别 vs 普通话识别的错误率之差（普通话识别也能听懂 → 可能是「塑料粤语」，试验性）
    音质    削波、结尾被截断

三、综合分
    默认权重；用户在「🎧 试听」打过分后可以「校准」：非负最小二乘拟合权重，留一法验证，
    只有比默认权重更贴近用户评分时才启用（保存在 workspace/_score_weights.json）。

用法（特征提取，GPT-SoVITS runtime）：python autoscore.py <job.json> <out.json> [--device cpu|cuda:0]
    job.json: {"items": [{"wav", "text", "lang", "voice"}]}
"""
import json
import math
import os
import re
import sys

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

DIMS = ["准确度", "音色", "停顿", "语速", "音高", "起伏", "音质"]
# 权重依据：用 20 条人工评分验证（2026-09-27），准确度、音色与人耳评分的秩相关 0.57 / 0.62；
# 语速、音高、起伏 ≈ 0（情绪高昂时音高本来就会升高），只作为说明，几乎不计分。「口音」维度实测 -0.33 且大量误报，已删除。
DEFAULT_W = {"准确度": 0.40, "音色": 0.35, "停顿": 0.10, "语速": 0.05, "音高": 0.03, "起伏": 0.02, "音质": 0.05}
PROFILE_VERSION = 2
PUNCT = "，。！？、；：,.!?;:…~～"
CN_DIGITS = "零一二三四五六七八九"
SR = 16000


# ------------------------------------------------------------------ 文本
_T2S = None


def _t2s(t):
    """繁体转简体（识别结果可能输出「著」等繁体/异体字）。"""
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:
            _T2S = False
    t = _T2S.convert(t) if _T2S else t
    # 常见异体字统一（原文和识别结果两边都转，不影响比较）
    return t.translate(_VARIANTS)


_VARIANTS = str.maketrans("著裏喫噉係佢哋", "着里吃咁系佢哋")


def norm_text(t):
    """去标点、繁转简、阿拉伯数字逐位转汉字（配合「不做数字转写」的识别结果对齐）。"""
    t = re.sub(r"\d", lambda m: CN_DIGITS[int(m.group(0))], _t2s(t or ""))
    return re.sub(r"[^一-鿿A-Za-z]", "", t).lower()


def edit_ops(ref, hyp):
    """(替换, 删除/漏读, 插入/多读)"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = range(n + 1)
    d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I


# ------------------------------------------------------------------ 声学特征
def acoustic(y):
    """语速计算用的有效时长、句中长停顿、音高、削波、截断。y: 16k 单声道 float。"""
    import librosa
    hop = SR // 100
    n = len(y) // hop
    if n < 10:
        return None
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    noise, peak = np.percentile(db, 10), np.percentile(db, 98)
    voiced = db > max(noise + 10, peak - 35)
    idx = np.where(voiced)[0]
    b, e = idx[0], idx[-1] + 1
    seg = voiced[b:e]
    pauses, run = [], 0
    for v in seg:
        if not v:
            run += 1
        else:
            if run >= 15:  # ≥0.15 秒算一次停顿
                pauses.append(run / 100)
            run = 0
    speech = (e - b) / 100 - sum(pauses)
    pause_ratio = sum(pauses) / max((e - b) / 100, 0.1)
    f0, vf, _ = librosa.pyin(y[b * hop:e * hop], fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=160)
    f0 = f0[vf] if vf is not None else np.array([])
    f0 = f0[~np.isnan(f0)]
    med = float(np.median(f0)) if len(f0) > 10 else None
    st_std = float(np.std(12 * np.log2(f0 / med))) if med else None
    tail = 20 * np.log10(np.sqrt((y[-int(0.03 * SR):] ** 2).mean()) + 1e-9)
    return {"speech": round(float(speech), 3), "pauses": pauses, "pause_ratio": round(pause_ratio, 3),
            "max_pause": max(pauses) if pauses else 0.0, "f0": med, "f0_std": st_std,
            "clip": float((np.abs(y) > 0.99).mean()), "truncated": bool(tail > peak - 12)}


# ------------------------------------------------------------------ 本人基准
def profile(pid, lang, embedder=None):
    """这个人某种语言的基准：语速、音高、起伏、声纹。缓存到 workspace/<pid>/profile_<lang>.json"""
    import core
    import diarize as D
    proj = core.load_project(pid)
    segs = [s for s in proj["segments"] if s["keep"] and s["lang"] == lang and s["text"].strip()][:40]
    if not segs:
        return None
    cache = os.path.join(core.project_dir(pid), f"profile_{lang}.json")
    key = f"v{PROFILE_VERSION}|" + "|".join(s["file"] for s in segs)
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("key") == key:
            c["vp"] = np.array(c["vp"])
            return c
    rates, f0s, stds, embs, pr = [], [], [], [], []
    for s in segs:
        y = D.load_audio(os.path.join(core.project_dir(pid), "slices", s["file"]))
        a = acoustic(y)
        if not a or a["speech"] <= 0.5:
            continue
        rates.append(len(norm_text(s["text"])) / a["speech"])
        pr.append(a["pause_ratio"])
        if a["f0"]:
            f0s.append(a["f0"])
            stds.append(a["f0_std"])
        embs.append(y)
    emb = (embedder or D.Embedder())(embs[:30])
    vp = emb.mean(0)
    vp = vp / np.linalg.norm(vp)
    c = {"key": key, "rate": float(np.median(rates)), "f0": float(np.median(f0s)), "f0_std": float(np.median(stds)),
         "pause_ratio": float(np.median(pr)),
         "vp": vp.tolist(), "n": len(rates)}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(c, f)
    c["vp"] = vp
    return c


# ------------------------------------------------------------------ 打分（纯计算，网页进程可直接用）
def clip100(x):
    return float(max(0.0, min(100.0, x)))


def subscores(ft, prof):
    """特征 → 各维度 0~100 分 + 具体原因。"""
    s, flags = {}, []
    ref = ft.get("ref_text", "")
    if ref:
        S, Dl, I = ft["ops"]
        err = (S + Dl + I) / max(len(ref), 1)
        s["准确度"] = clip100(100 * (1 - err))
        if Dl:
            flags.append(f"漏读约 {Dl} 字")
        if S + I:
            flags.append(f"读错/多读约 {S + I} 字（识别为：{ft['asr'][:30]}）")
    a = ft.get("ac")
    if a and prof:
        if ft.get("n_chars") and a["speech"] > 0.3:
            r = ft["n_chars"] / a["speech"] / prof["rate"]
            s["语速"] = clip100(100 - 200 * abs(math.log(r)))
            # 实测语速与人工评分不相关（-0.15），只作说明
            if r > 1.3:
                flags.append(f"ℹ️ 语速比本人素材快 {int((r - 1) * 100)}%")
            elif r < 0.77:
                flags.append(f"ℹ️ 语速比本人素材慢 {int((1 - r) * 100)}%")
        # 实测：「停顿太多」的那条最长停顿 0.57 秒、停顿占比 25%，其余 ≤0.32 秒、15~21%
        long_pen = 80 * max(0.0, a["max_pause"] - 0.4)
        base_ratio = prof.get("pause_ratio", 0.18)
        ratio_pen = 150 * max(0.0, a["pause_ratio"] - base_ratio * 1.15)
        s["停顿"] = clip100(100 - long_pen - ratio_pen)
        if a["max_pause"] >= 0.45:
            flags.append(f"有明显的长停顿（最长 {a['max_pause']:.2f} 秒）")
        if ratio_pen >= 5:
            flags.append(f"停顿偏多（占 {a['pause_ratio'] * 100:.0f}%，本人约 {base_ratio * 100:.0f}%）")
        if a["f0"] and prof.get("f0"):
            d = 12 * math.log2(a["f0"] / prof["f0"])
            s["音高"] = clip100(100 - 15 * abs(d))
            if abs(d) >= 3:
                flags.append(f"ℹ️ 音高比本人素材{'高' if d > 0 else '低'} {abs(d):.1f} 个半音（情绪高昂时偏高是正常的）")
        if a["f0_std"] and prof.get("f0_std"):
            ratio = a["f0_std"] / prof["f0_std"]
            s["起伏"] = clip100(100 - 120 * abs(math.log(ratio)))
            if ratio < 0.6:
                flags.append(f"ℹ️ 语调偏平（起伏约为本人的 {int(ratio * 100)}%）")
            elif ratio > 1.6:
                flags.append("ℹ️ 语调起伏比本人大")
        s["音质"] = clip100(100 - min(60, a["clip"] * 20000) - (30 if a["truncated"] else 0))
        if a["clip"] > 0.001:
            flags.append("有爆音")
        if a["truncated"]:
            flags.append("结尾像被截断")
    if ft.get("sim") is not None:
        s["音色"] = clip100((ft["sim"] - 0.4) / 0.45 * 100)
        if a and a["speech"] < 3.0:
            s["_短句"] = 1  # 3 秒以内的短句声纹不可靠（同类短句音色分 19~100 不等），综合分里减半
            flags.append("ℹ️ 句子太短，音色分参考价值低")
    if ft.get("sim") is not None and ft["sim"] < 0.55 and not (a and a["speech"] < 3.0):
        flags.append("音色和本人差别较大")
    return s, flags


def load_weights():
    import core
    p = os.path.join(core.WORK, "_score_weights.json")
    try:
        with open(p, encoding="utf-8") as f:
            w = json.load(f)
        return w["weights"], w
    except Exception:
        return dict(DEFAULT_W), None


def composite(s, w):
    w = dict(w)
    if s.get("_短句"):
        w["音色"] = w.get("音色", 0) / 2
    keys = [k for k in s if not k.startswith("_")]
    tot = sum(w.get(k, 0) for k in keys)
    return round(sum(s[k] * w.get(k, 0) for k in keys) / tot, 1) if tot else None


def stars(score):
    return None if score is None else round(1 + 4 * score / 100, 1)


def calibrate(samples):
    """samples: [(子分数 dict, 用户星级)]。非负最小二乘拟合权重，留一法验证。返回 (权重或 None, 报告)。"""
    from scipy.optimize import nnls
    from scipy.stats import spearmanr
    dims = [d for d in DIMS if sum(1 for s, _ in samples if d in s) >= max(3, len(samples) * 0.6)]
    rows = [(s, y) for s, y in samples if all(d in s for d in dims)]
    if len(rows) < 15:
        return None, f"有效评分只有 {len(rows)} 条，至少需要 15 条才能校准（样本太少容易过拟合），先用默认权重"
    X = np.array([[s[d] / 100 for d in dims] + [1.0] for s, _ in rows])
    y = np.array([float(v) for _, v in rows])
    base = [composite(s, DEFAULT_W) for s, _ in rows]
    rho_default = spearmanr(base, y).correlation
    loo = []
    for i in range(len(rows)):
        m = np.ones(len(rows), bool)
        m[i] = False
        coef, _ = nnls(X[m], y[m])
        loo.append(float(X[i] @ coef))
    rho_loo = spearmanr(loo, y).correlation
    coef, _ = nnls(X, y)
    w = {d: float(c) for d, c in zip(dims, coef[:-1]) if c > 1e-6}
    report = (f"{len(rows)} 条评分：默认权重与你的评分秩相关 {rho_default:.2f}；"
              f"校准后（留一法验证）{rho_loo:.2f}")
    if not w or np.isnan(rho_loo) or rho_loo <= (rho_default if not np.isnan(rho_default) else -1) + 0.10:
        return None, report + "。校准没有明显更好，继续用默认权重。"
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}, report + "。已启用校准后的权重。"


# ------------------------------------------------------------------ 网页/脚本调用入口
def item_info(path):
    """找出音频对应的原文、语言、声音：先查生成记录（outputs/_takes），再查同目录的 文案.json。"""
    import core
    import takes
    ap = os.path.abspath(path)
    for tid in os.listdir(takes.TAKES):
        mp = os.path.join(takes.TAKES, tid, "meta.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if os.path.abspath(m.get("wav", "")) == ap:
            return {"text": "".join(s["text"] for s in m["sentences"]), "lang": takes._lang(m), "voice": m["vid"]}
    side = os.path.join(os.path.dirname(ap), "文案.json")
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            d = json.load(f)
        info = d.get(os.path.basename(ap)) or d.get("*")
        if info:
            names = {n: p for n, p in core.list_projects()}
            return {"text": info.get("text", ""), "lang": info.get("lang", "zh"),
                    "voice": names.get(info.get("voice"), info.get("voice"))}
    return None


def evaluate(paths, log=print, device="auto"):
    """对一批音频自动评分。返回 [{"wav","score","stars","sub","flags","asr"}]；找不到原文的只评声学部分。"""
    import core
    import gpu
    items, keep = [], []
    for p in paths:
        info = item_info(p) or {}
        items.append({"wav": p, "text": info.get("text", ""), "lang": info.get("lang", "zh"), "voice": info.get("voice")})
    if device == "auto":
        used, total = gpu.vram()
        device = "cuda:0" if (not gpu.holder() and total - used > 2500) else "cpu"
    job = os.path.join(core.WORK, f"_score_job_{os.getpid()}.json")
    out = job.replace("_job_", "_out_")
    with open(job, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False)
    log(f"自动评分：{len(items)} 个音频（{'显卡' if device != 'cpu' else 'CPU'}）…")
    code = core.run_cmd([os.path.join(APP_DIR, "autoscore.py"), job, out, "--device", device],
                        lambda s: log(s[8:]) if s.startswith("[score]") else (log("  " + s[-200:]) if "rror" in s else None))
    os.remove(job)
    if code != 0 or not os.path.exists(out):
        raise RuntimeError("自动评分失败，请查看日志")
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    os.remove(out)
    w, _ = load_weights()
    for it, r in zip(items, res):
        s, flags = subscores(r["ft"], r["profile"])
        sc = composite(s, w)
        keep.append({"wav": it["wav"], "score": sc, "stars": stars(sc), "sub": {k: round(v) for k, v in s.items()},
                     "flags": flags, "asr": r["ft"].get("asr", ""), "has_text": bool(it["text"])})
    return keep


# ------------------------------------------------------------------ 特征提取主程序（子进程）
def main():
    import argparse
    import shutil
    import tempfile
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device == "cpu":
        torch.cuda.is_available = lambda: False
    import core
    import diarize as D
    import textnorm
    with open(a.job, encoding="utf-8") as f:
        items = json.load(f)["items"]
    tmp = tempfile.mkdtemp(prefix="score_", dir=core.WORK)
    for i, it in enumerate(items):
        y = D.load_audio(it["wav"])
        from scipy.io import wavfile
        wavfile.write(os.path.join(tmp, f"{i:04d}.wav"), SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    log = lambda m: print("[score]", m, flush=True)
    asr = {}
    for lang in {it.get("lang", "zh") for it in items}:
        asr[lang] = core.run_sensevoice_raw(tmp, lang, log, device=a.device)
    emb = D.Embedder()
    profs = {}
    out = []
    for i, it in enumerate(items):
        name = f"{i:04d}.wav"
        y = D.load_audio(it["wav"])
        ac = acoustic(y)
        lang = it.get("lang", "zh")
        key = (it.get("voice"), lang)
        if key not in profs and it.get("voice"):
            try:
                profs[key] = profile(it["voice"], lang, emb)
            except Exception as e:
                log(f"基准计算失败：{e}")
                profs[key] = None
        prof = profs.get(key)
        ft = {"wav": it["wav"], "ac": ac, "asr": asr.get(lang, {}).get(name, "")}
        text = it.get("text") or ""
        if text:
            spoken = textnorm.spoken(text, lang)
            ref, hyp = norm_text(spoken), norm_text(ft["asr"])
            ft.update(ref_text=ref, ops=edit_ops(ref, hyp), n_chars=len(ref),
                      n_punct=sum(1 for ch in text.strip().rstrip(PUNCT) if ch in PUNCT))
        if prof is not None:
            ft["sim"] = float(emb([y])[0] @ prof["vp"])
        prof_out = {k: v for k, v in (prof or {}).items() if k != "vp"}
        out.append({"ft": ft, "profile": prof_out})
    shutil.rmtree(tmp, ignore_errors=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
