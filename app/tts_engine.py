# -*- coding: utf-8 -*-
"""文字转语音：按情感分句，为每句挑选同情感的参考音频，让语气自然有起伏。"""
import gc
import os
import random
import re
import time

import numpy as np

import core

TEXT_LANGS = {
    "普通话（可夹英文）": "zh",
    "粤语（可夹英文）": "yue",
    "自动识别（普通话/英文混合）": "auto",
    "自动识别（粤语/英文混合）": "auto_yue",
}

# 自动情感：按关键词和标点判断每句话的情绪
EMO_RULES = [
    ("生气", r"气死|可恶|混蛋|凭什么|滚|闭嘴|太过分|受够|烦死|岂有此理|嬲|顶你|激死"),
    ("悲伤", r"唉|可惜|难过|伤心|遗憾|对不起|哭|想念|舍不得|失去|再也|心痛|惨|唔开心|好攰"),
    ("温柔", r"晚安|宝贝|别怕|没关系|慢慢来|辛苦了|抱抱|乖|放心|唔紧要|慢慢嚟"),
    ("惊讶", r"什么[？?!！]|真的吗|天哪|不会吧|竟然|居然|怎么可能|咩话|真係|唔係啩|吓[？?]"),
    ("开心", r"哈哈|太好了|开心|高兴|棒|耶|恭喜|好开心|谢谢|喜欢|好嘢|正呀|劲"),
    ("严肃", r"注意|必须|禁止|警告|严禁|务必|切记|请勿"),
]
# 没有某种情感的参考录音时，按顺序找相近的情感（实测：「兴奋」句用「开心」录音比用「平静」录音像得多）
SIMILAR_EMO = {
    "兴奋": ["开心", "惊讶"], "开心": ["兴奋"], "惊讶": ["兴奋", "开心"], "温柔": ["平静", "悲伤"],
    "悲伤": ["温柔"], "生气": ["严肃", "兴奋"], "严肃": ["平静", "生气"], "平静": ["温柔", "严肃"],
}
TAG_RE = re.compile(r"[\[【]\s*(" + "|".join(core.EMOTIONS) + r")\s*[\]】]")
SENT_RE = re.compile(r"(?<=[。！？!?…~～\n])")


def detect_emotion(sent, default):
    for emo, pat in EMO_RULES:
        if re.search(pat, sent):
            return emo
    if re.search(r"[!！]{2,}|[!！]$", sent.strip()):
        return "兴奋"
    return default


def split_by_emotion(text, default, auto):
    """解析 [开心] 这类标签；未标注的部分按句自动判断（可关闭）。返回 [(情感, 文本)]，相邻同情感合并。"""
    parts, cur, pos = [], None, 0
    for m in TAG_RE.finditer(text):
        if m.start() > pos:
            parts.append((cur, text[pos:m.start()]))
        cur, pos = m.group(1), m.end()
    parts.append((cur, text[pos:]))
    out = []
    for tag, chunk in parts:
        for sent in SENT_RE.split(chunk):
            if not sent.strip():
                continue
            emo = tag or (detect_emotion(sent, default) if auto else default)
            if out and out[-1][0] == emo:
                out[-1] = (emo, out[-1][1] + sent)
            else:
                out.append((emo, sent))
    return out


LONG_SPLIT_RE = re.compile(r"(?<=[，,；;：:])")


def split_sentences(text, default, auto, max_len=60):
    """逐句拆分（不合并），用于逐句生成和字幕：[(情感, 句子)]。超长句在逗号处再拆。"""
    out = []
    for emo, chunk in split_by_emotion_raw(text, default, auto):
        if len(chunk) <= max_len:
            out.append((emo, chunk))
            continue
        buf = ""
        for part in LONG_SPLIT_RE.split(chunk):
            if buf and len(buf) + len(part) > max_len:
                out.append((emo, buf))
                buf = ""
            buf += part
        if buf.strip():
            out.append((emo, buf))
    return [(e, s.strip()) for e, s in out if s.strip()]


def split_by_emotion_raw(text, default, auto):
    parts, cur, pos = [], None, 0
    for m in TAG_RE.finditer(text):
        if m.start() > pos:
            parts.append((cur, text[pos:m.start()]))
        cur, pos = m.group(1), m.end()
    parts.append((cur, text[pos:]))
    out = []
    for tag, chunk in parts:
        for sent in SENT_RE.split(chunk):
            if sent.strip():
                out.append((tag or (detect_emotion(sent, default) if auto else default), sent))
    return out


class Engine:
    def __init__(self):
        self.tts = None
        self.loaded = (None, None)

    def unload(self):
        if self.tts is not None:
            self.tts = None
            self.loaded = (None, None)
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def _ensure(self, voice):
        if voice["trained"]:
            sovits = os.path.join(voice["dir"], "sovits.pth")
            gpt = os.path.join(voice["dir"], "gpt.ckpt")
        else:
            sovits = os.path.join(core.GSV, core.PRE_S2G)
            gpt = os.path.join(core.GSV, core.PRE_S1)
        if self.tts is None:
            from TTS_infer_pack.TTS import TTS, TTS_Config
            cfg = TTS_Config({"version": "v2", "custom": {
                "device": "cuda", "is_half": True, "version": "v2",
                "t2s_weights_path": gpt, "vits_weights_path": sovits,
                "bert_base_path": core.BERT_DIR, "cnhuhbert_base_path": core.HUBERT_DIR}})
            self.tts = TTS(cfg)
            self.loaded = (sovits, gpt)
            return
        if self.loaded[0] != sovits:
            self.tts.init_vits_weights(sovits)
        if self.loaded[1] != gpt:
            self.tts.init_t2s_weights(gpt)
        self.loaded = (sovits, gpt)

    @staticmethod
    def pick_refs(voice, emotion, text_lang, rng, max_dur=None):
        """挑参考录音。语言优先于情感（用普通话录音生成粤语会带口音）：
        锁定的参考 → 同语言同情感 → 同语言相近情感（如 兴奋→开心）→ 同语言平静 → 同语言其他 → 其他语言同情感 → 其他。
        voice.json 里的参考已按质量排序，只在最好的两条之间随机（让「重做」有变化）；锁定的参考固定使用。"""
        refs = [r for r in voice["refs"] if not max_dur or r["dur"] <= max_dur] or voice["refs"]
        want = "yue" if "yue" in text_lang else ("zh" if text_lang in ("zh", "auto") else "en")
        pinned = (voice.get("pinned") or {}).get(f"{want}|{emotion}")
        pin = [r for r in refs if r["file"] == pinned]
        if pin:
            return pin[0], [r for r in refs if r["lang"] == want and r["emotion"] == pin[0]["emotion"] and r is not pin[0]][:3], True
        same = [r for r in refs if r["lang"] == want]
        tiers = [[r for r in same if r["emotion"] == emotion]]
        tiers += [[r for r in same if r["emotion"] == e] for e in SIMILAR_EMO.get(emotion, [])]
        tiers += [[r for r in same if r["emotion"] == "平静"], same, [r for r in refs if r["emotion"] == emotion], refs]
        c = next(t for t in tiers if t)
        first = c[rng.randrange(min(2, len(c)))]
        aux = [r for r in c if r is not first][:3]
        return first, aux, (first["emotion"] == emotion)

    def synthesize(self, vid, text, lang_label, default_emo, auto_emo, expressiveness, speed, pause,
                   seed, multi_ref, log=print):
        voice = core.load_voice(vid)
        self._ensure(voice)
        text_lang = TEXT_LANGS[lang_label]
        seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2 ** 31 - 1)
        rng = random.Random(seed)
        x = float(expressiveness)
        groups = split_by_emotion(text.strip(), default_emo, auto_emo)
        if not groups:
            raise ValueError("请输入要朗读的文字")
        sr, pieces, report = 32000, [], []
        for i, (emo, seg) in enumerate(groups):
            ref, aux, hit = self.pick_refs(voice, emo, text_lang, rng, max_dur=core.REF_MAX)  # GPT-SoVITS 限 3~10 秒
            ref_path = os.path.join(voice["dir"], "refs", ref["file"])
            inputs = {
                "text": seg, "text_lang": text_lang,
                "ref_audio_path": ref_path, "prompt_text": ref["text"], "prompt_lang": ref["lang"],
                "aux_ref_audio_paths": [os.path.join(voice["dir"], "refs", r["file"]) for r in aux] if multi_ref else [],
                "top_k": int(5 + 25 * x), "top_p": round(0.75 + 0.25 * x, 2), "temperature": round(0.6 + 0.7 * x, 2),
                "text_split_method": "cut5", "batch_size": 1, "speed_factor": float(speed),
                "fragment_interval": float(pause), "seed": seed + i, "parallel_infer": True,
                "repetition_penalty": 1.35, "split_bucket": False,
            }
            for sr, audio in self.tts.run(inputs):
                pieces.append(audio.astype(np.int16))
                pieces.append(np.zeros(int(sr * float(pause)), dtype=np.int16))
            report.append(f"{i + 1}. 【{emo}{'' if hit else '→无此情感素材，用' + ref['emotion']}】{seg.strip()[:40]}")
        audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.int16)
        from scipy.io import wavfile
        name = re.sub(r'[\\/:*?"<>|\s]', "", voice["name"])[:20]
        out = os.path.join(core.OUTPUTS, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.wav")
        wavfile.write(out, sr, audio)
        return out, seed, "\n".join(report)


ENGINE = Engine()
