# -*- coding: utf-8 -*-
"""SenseVoice 语音识别（粤语/普通话/英语，带标点，附带情感与语种）。voxcpm_env 运行：

    python sensevoice_asr.py <wav目录或文件> <输出json> [--lang zh|yue|en|auto]

输出 {文件名: {"text": 文字, "emotion": 中文情感, "lang": 识别语种}}
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "SenseVoiceSmall")
EMO = {"HAPPY": "开心", "SAD": "悲伤", "ANGRY": "生气", "NEUTRAL": "平静", "FEARFUL": "悲伤",
       "DISGUSTED": "生气", "SURPRISED": "惊讶"}
TAG = re.compile(r"<\|([^|]+)\|>")
DEVICE = "cuda:0"


VAD = os.path.join(os.path.dirname(HERE), "GPT-SoVITS", "tools", "asr", "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch")


def timestamps(inp, out, lang):
    """整条长音频：VAD 分块 + 逐字时间戳。输出 {"chars": [[字, 开始秒, 结束秒], ...]}（含标点，标点时间为其位置）。"""
    from funasr import AutoModel
    model = AutoModel(model=MODEL, disable_update=True, device=DEVICE, vad_model=VAD,
                      vad_kwargs={"max_single_segment_time": 25000})
    r = model.generate(input=inp, language=lang, use_itn=True, output_timestamp=True, merge_vad=False)[0]
    chars = [[w, t[0] / 1000, t[1] / 1000] for w, t in zip(r.get("words") or [], r.get("timestamp") or [])
             if not TAG.fullmatch(w)]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False)
    print(f"[{len(chars)} chars]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 或 cpu（显卡被训练占用时用 cpu）")
    ap.add_argument("--no-itn", action="store_true", help="不把读出的数字转写成阿拉伯数字（评分时用，才能发现数字读错）")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    if a.timestamps:
        return timestamps(a.inp, a.out, a.lang)
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    files = ([os.path.join(a.inp, f) for f in sorted(os.listdir(a.inp)) if f.lower().endswith(".wav")]
             if os.path.isdir(a.inp) else [a.inp])
    model = AutoModel(model=MODEL, disable_update=True, device=DEVICE)
    res = {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        outs = model.generate(input=batch, language=a.lang, use_itn=not a.no_itn, batch_size=len(batch))
        for f, o in zip(batch, outs):
            raw = o["text"]
            tags = TAG.findall(raw)
            res[os.path.basename(f)] = {
                "text": rich_transcription_postprocess(raw).strip(),
                "emotion": next((EMO[t] for t in tags if t in EMO), "平静"),
                "lang": next((t for t in tags if t in ("zh", "yue", "en", "ja", "ko")), ""),
            }
        print(f"[{min(i + 16, len(files))}/{len(files)}]", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
