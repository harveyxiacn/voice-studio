# -*- coding: utf-8 -*-
"""用 emotion2vec 给切片自动标注情感。用法：python emotion_tag.py <wav目录> <输出json>"""
import json
import os
import sys

APP = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(APP, "models", "emotion2vec_plus_base")
MAP = {"neutral": "平静", "happy": "开心", "sad": "悲伤", "angry": "生气",
       "surprised": "惊讶", "fearful": "悲伤", "disgusted": "生气"}


def main(wav_dir, out):
    from funasr import AutoModel
    model = AutoModel(model=MODEL, disable_update=True, device="cuda:0")
    files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    res = {}
    for i, f in enumerate(files):
        r = model.generate(os.path.join(wav_dir, f), granularity="utterance", extract_embedding=False)[0]
        scores = {lab.split("/")[-1]: float(s) for lab, s in zip(r["labels"], r["scores"])}
        best = max(scores, key=scores.get)
        # 置信度不高时按平静处理，避免误标
        emo = MAP.get(best, "平静") if scores[best] >= 0.55 else "平静"
        # 语气明显高昂的“开心”标为兴奋
        if emo == "开心" and scores[best] >= 0.9 and scores.get("surprised", 0) > 0.03:
            emo = "兴奋"
        res[f] = emo
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}]", flush=True)
    with open(out, "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
