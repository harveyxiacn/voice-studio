# -*- coding: utf-8 -*-
"""朗读前的文本规范化：数字读法 + 发音词典（只影响读音，字幕仍显示原文）。

- 年份（1500~2099）逐位读：2026 → 二零二六；后面紧跟计量单位时（2026元、2026个）按数值读，不处理
- 5 位以上的长数字（电话、编号）逐位读
- 发音词典：用同音字替换纠正读音，例如粤语「珊姐」的「姐」要读上扬调，可写成「珊者」
  词典保存在 workspace/_pron_dict.json，按语言分：{"yue": {...}, "zh": {...}, "all": {...}}
"""
import json
import os
import re

import core

DICT_PATH = os.path.join(core.WORK, "_pron_dict.json")
DIGITS = "零一二三四五六七八九"
# 数字后面紧跟这些字时，是数量而不是年份，不逐位读
UNITS = set("元块塊个個人米克斤岁歲%％号號分秒名位次条條只隻件本张張台部辆輛万萬亿億千百倍公里吨噸")


def load_dict():
    try:
        with open(DICT_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return {k: dict(v) for k, v in d.items()}
    except Exception:
        return {}


def save_dict(d):
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def dict_rows():
    """给界面表格：[[原文, 读成, 语言], ...]"""
    names = {"yue": "粤语", "zh": "普通话", "all": "全部"}
    return [[k, v, names.get(lang, lang)] for lang, m in load_dict().items() for k, v in m.items()]


def save_rows(rows):
    codes = {"粤语": "yue", "普通话": "zh", "全部": "all", "": "all"}
    d = {}
    for r in rows or []:
        if not r or len(r) < 2 or not str(r[0]).strip() or not str(r[1]).strip():
            continue
        lang = codes.get(str(r[2]).strip() if len(r) > 2 and r[2] else "", "all")
        d.setdefault(lang, {})[str(r[0]).strip()] = str(r[1]).strip()
    save_dict(d)
    return sum(len(v) for v in d.values())


def _digits(s):
    return "".join(DIGITS[int(c)] for c in s)


def spoken(text, lang):
    """把要显示的文字转成要朗读的文字。lang：'yue' 或 'zh'。"""
    d = load_dict()
    entries = {**d.get("all", {}), **d.get(lang, {})}
    for k in sorted(entries, key=len, reverse=True):  # 长词优先，避免被短词拆掉
        text = text.replace(k, entries[k])

    def year(m):
        nxt = m.string[m.end():m.end() + 1]
        return m.group(0) if nxt and nxt in UNITS else _digits(m.group(0))

    text = re.sub(r"(?<![\d.])(1[5-9]\d{2}|20\d{2})(?![\d.])", year, text)
    text = re.sub(r"(?<![\d.])\d{5,}(?![\d.])", lambda m: _digits(m.group(0)), text)
    return text
