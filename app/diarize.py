# -*- coding: utf-8 -*-
"""说话人分离：自动判断音频/视频里有几个人，并给出每个人的发言时间段。

用法（GPT-SoVITS runtime，工作目录为 GPT-SoVITS）：
    python diarize.py <输出json> <音频或视频> [更多文件...] [--threshold 0.55] [--num N]

流程：FSMN-VAD 找人声 → 1.5 秒滑窗提取 ERes2NetV2 声纹 → 层次聚类自动定人数
     → 合并过于相似的类、并入零碎类 → 逐帧投票得到发言段。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

GSV = os.getcwd()
sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS"), os.path.join(GSV, "GPT_SoVITS", "eres2net"), os.path.join(GSV, "tools")]

SR = 16000
WIN, HOP = 1.5, 0.5          # 声纹窗口长度 / 步长（秒）
MIN_SEG = 0.4                # 短于此的人声段不参与聚类，事后按邻近归属
FRAME = 0.01


def log(*a):
    print("[diar]", *a, flush=True)


def load_audio(path):
    from tools.my_utils import load_audio as la
    return la(path, SR).astype(np.float32)


def vad(wav):
    from funasr import AutoModel
    m = AutoModel(model=os.path.join(GSV, "tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                  model_revision="v2.0.4", disable_update=True)
    segs = m.generate(input=wav, fs=SR)[0]["value"]
    return [(b / 1000, e / 1000) for b, e in segs]


class Embedder:
    def __init__(self):
        from ERes2NetV2 import ERes2NetV2
        import kaldi as Kaldi
        self.K = Kaldi
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        m.load_state_dict(torch.load(os.path.join(GSV, "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
                                     map_location="cpu", weights_only=False))
        self.m = m.eval().to(self.dev)

    @torch.no_grad()
    def __call__(self, chunks, bs=64):
        out = []
        # 按长度分组成批（同批等长），短于 1 秒的片段循环补齐到 1 秒
        chunks = [np.tile(c, int(np.ceil(SR / max(len(c), 1))))[:max(len(c), SR)] for c in chunks]
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        emb = [None] * len(chunks)
        i = 0
        while i < len(order):
            L = len(chunks[order[i]])
            grp = [k for k in order[i:i + bs] if len(chunks[k]) == L]
            feats = []
            for k in grp:
                f = self.K.fbank(torch.from_numpy(chunks[k]).unsqueeze(0), num_mel_bins=80, sample_frequency=SR, dither=0)
                feats.append(f - f.mean(0, keepdim=True))
            e = self.m(torch.stack(feats).to(self.dev)).float().cpu().numpy()
            for k, v in zip(grp, e):
                emb[k] = v
            i += len(grp)
        out = [np.stack(emb)]
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def windows(segs):
    wins = []
    for b, e in segs:
        L = e - b
        if L < MIN_SEG:
            continue
        if L <= WIN:
            wins.append((b, e))
            continue
        t = b
        while t + WIN < e:
            wins.append((t, t + WIN))
            t += HOP
        wins.append((e - WIN, e))
    return wins


def cluster(emb, durs, threshold, num=None):
    from scipy.cluster.hierarchy import fcluster, linkage
    n = len(emb)
    if n < 3:
        return np.zeros(n, int)
    Z = linkage(emb, method="average", metric="cosine")
    lab = fcluster(Z, t=num, criterion="maxclust") if num else fcluster(Z, t=threshold, criterion="distance")
    lab = np.unique(lab, return_inverse=True)[1]

    def cents(lab):
        ks = np.unique(lab)
        c = np.stack([emb[lab == k].mean(0) for k in ks])
        return ks, c / np.linalg.norm(c, axis=1, keepdims=True)

    total = durs.sum()
    for _ in range(20):
        ks, c = cents(lab)
        if len(ks) == 1:
            break
        # 1) 零碎类（时长 < 5% 且 < 4 秒）并入最近的类
        size = np.array([durs[lab == k].sum() for k in ks])
        small = [i for i in range(len(ks)) if size[i] < max(0.05 * total, 0) and size[i] < 4.0 and not num]
        if small:
            i = small[int(np.argmin(size[small]))]
            sims = c @ c[i]
            sims[i] = -9
            lab[lab == ks[i]] = ks[int(np.argmax(sims))]
            continue
        # 2) 质心非常接近的两类合并（一个人被拆成两类）
        S = c @ c.T
        np.fill_diagonal(S, -9)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        # 质心相似度 > 1 - 1.05×阈值（默认 0.55 → 0.42）即视为同一人。
        # 标定：合成三人对话不同人之间 ≤0.20；真实户外粤语 vlog 同一人两类之间 0.48~0.50。
        if S[i, j] > 1 - threshold * 1.05 and not num:
            lab[lab == ks[j]] = ks[i]
            continue
        break
    # 3) 按最近质心重新分配一次，修正边界窗口
    for _ in range(2):
        ks, c = cents(lab)
        lab = ks[np.argmax(emb @ c.T, axis=1)]
    return np.unique(lab, return_inverse=True)[1]


def to_turns(segs, wins, lab, n_spk, dur, trim=0.15):
    """逐帧投票：每 10ms 帧取覆盖它、且中心最近的窗口标签。"""
    nf = int(dur / FRAME) + 1
    frame_lab = np.full(nf, -1)
    centers = np.array([(a + b) / 2 for a, b in wins])
    for b, e in segs:
        fb, fe = int(b / FRAME), min(int(e / FRAME), nf)
        if fe <= fb or not len(wins):
            continue
        t = (np.arange(fb, fe) + 0.5) * FRAME
        idx = np.abs(t[:, None] - centers[None, :]).argmin(1)
        frame_lab[fb:fe] = lab[idx]
    # 平滑：短于 0.3 秒的插段并入前一段
    turns = []
    i = 0
    while i < nf:
        if frame_lab[i] < 0:
            i += 1
            continue
        j = i
        while j < nf and frame_lab[j] == frame_lab[i]:
            j += 1
        turns.append([i * FRAME, j * FRAME, int(frame_lab[i])])
        i = j
    merged = []
    for t in turns:
        if merged and t[2] != merged[-1][2] and t[1] - t[0] < 0.3 and t[0] - merged[-1][1] < 0.05:
            merged[-1][1] = t[1]
            continue
        if merged and t[2] == merged[-1][2] and t[0] - merged[-1][1] < 0.05:
            merged[-1][1] = t[1]
            continue
        merged.append(t)
    # 换人交界处各裁掉一点，减少串音
    out = []
    for k, (b, e, s) in enumerate(merged):
        if k > 0 and merged[k - 1][2] != s and b - merged[k - 1][1] < 0.1:
            b += trim
        if k + 1 < len(merged) and merged[k + 1][2] != s and merged[k + 1][0] - e < 0.1:
            e -= trim
        if e - b >= 0.3:
            out.append([round(b, 2), round(e, 2), s])
    return out


MAX_CLUSTER = 6000  # 窗口太多（约 50 分钟以上）时先抽样聚类，再按质心分配其余窗口


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+", help="一个或多个音频/视频文件（联合识别，跨文件同一人会归为同一说话人）")
    ap.add_argument("--threshold", type=float, default=0.55, help="聚类距离阈值（越大越倾向于合并成更少的人）")
    ap.add_argument("--num", type=int, default=0, help="已知人数时直接指定")
    a = ap.parse_args()
    files, all_wins, all_chunks = [], [], []
    for fi, path in enumerate(a.inputs):
        wav = load_audio(path)
        dur = len(wav) / SR
        segs = vad(wav)
        wins = windows(segs)
        speech = sum(e - b for b, e in segs)
        log(f"{os.path.basename(path)}：时长 {dur:.1f} 秒，人声 {speech:.1f} 秒")
        files.append({"path": path, "duration": round(dur, 2), "speech": round(speech, 2), "segs": segs, "wins": wins})
        all_wins += [(fi, b, e) for b, e in wins]
        all_chunks += [wav[int(b * SR):int(e * SR)] for b, e in wins]
    res = {"files": [{k: f[k] for k in ("path", "duration", "speech")} for f in files], "speakers": [], "turns": []}
    if not all_wins:
        log("没有检测到人声")
    else:
        log(f"提取声纹：{len(all_wins)} 个窗口")
        emb = Embedder()(all_chunks)
        durs = np.array([e - b for _, b, e in all_wins]) * (HOP / WIN)
        if len(emb) > MAX_CLUSTER:
            idx = np.random.default_rng(0).choice(len(emb), MAX_CLUSTER, replace=False)
            sub = cluster(emb[idx], durs[idx], a.threshold, a.num or None)
            c = np.stack([emb[idx][sub == k].mean(0) for k in range(sub.max() + 1)])
            lab = np.argmax(emb @ (c / np.linalg.norm(c, axis=1, keepdims=True)).T, axis=1)
        else:
            lab = cluster(emb, durs, a.threshold, a.num or None)
        n = int(lab.max()) + 1
        turns = []
        for fi, f in enumerate(files):
            sel = [k for k, w in enumerate(all_wins) if w[0] == fi]
            if sel:
                turns += [[fi] + t for t in to_turns(f["segs"], f["wins"], lab[sel], n, f["duration"])]
        # 按说话总时长排序，编号从 0 开始；去掉没有任何发言段的类
        tot = [sum(t[2] - t[1] for t in turns if t[3] == k) for k in range(n)]
        order = [k for k in sorted(range(n), key=lambda k: -tot[k]) if tot[k] > 0]
        remap = {k: i for i, k in enumerate(order)}
        turns = [[fi, b, e, remap[s]] for fi, b, e, s in turns if s in remap]
        C = np.stack([emb[lab == k].mean(0) for k in order])
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        res["turns"] = turns
        res["similarity"] = np.round(C @ C.T, 3).tolist()
        res["centroids"] = np.round(C, 5).tolist()
        for i in range(len(order)):
            ts = [t for t in turns if t[3] == i]
            res["speakers"].append({"id": i, "dur": round(sum(t[2] - t[1] for t in ts), 2), "n_turns": len(ts)})
        log(f"识别到 {len(order)} 位说话人：" + "，".join(f"说话人{s['id'] + 1} {s['dur']:.0f}秒" for s in res["speakers"]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


def voiceprint(out, files):
    """声音项目的声纹：多段切片声纹的平均（最多取 80 段），保存为 .npy。"""
    import random
    files = list(files)
    random.Random(0).shuffle(files)
    chunks = [load_audio(f) for f in files[:80]]
    emb = Embedder()([c for c in chunks if len(c) > SR * 0.8])
    c = emb.mean(0)
    np.save(out, c / np.linalg.norm(c))
    log(f"声纹已更新（{len(emb)} 段）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "voiceprint":
        voiceprint(sys.argv[2], sys.argv[3:])
    else:
        main()
