# -*- coding: utf-8 -*-
"""VoxCPM2 LoRA 微调（8GB 显存版）。用 voxcpm_env 运行：

    python voxcpm_lora_train.py <项目ID> [--iters 800] [--rank 32] [--lr 1e-4] [--accum 8]

读取 workspace/<项目ID>/project.json 中「保留且有文稿」的切片作为训练集，
LoRA 权重保存在 workspace/<项目ID>/voxcpm_lora/step_xxxxxxx/。

省显存做法：官方训练模式用 fp32 加载 20 亿参数（约 8GB），这里把冻结的底座权重转为 bf16，
只有 LoRA 参数保持 fp32；训练本身就在 bf16 autocast 下进行，效果影响很小。
"""
import argparse
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "voxcpm_src")
MODEL = os.path.join(HERE, "models", "VoxCPM2")


def build(pid, a):
    pdir = os.path.join(ROOT, "workspace", pid)
    with open(os.path.join(pdir, "project.json"), encoding="utf-8") as f:
        proj = json.load(f)
    segs = [s for s in proj["segments"] if s["keep"] and s["text"].strip()]
    if not segs:
        sys.exit("没有可用于训练的片段（需要保留且有文稿）")
    out = os.path.join(pdir, "voxcpm_lora")
    # 官方脚本会从已有检查点续训；每次重新训练都从头开始，先清掉上一次的检查点
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    manifest = os.path.join(out, "train.jsonl")
    with open(manifest, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps({"audio": os.path.join(pdir, "slices", s["file"]).replace("\\", "/"),
                                "text": s["text"].strip(), "duration": s["dur"]}, ensure_ascii=False) + "\n")
    total = sum(s["dur"] for s in segs)
    print(f"训练集：{len(segs)} 段，{total / 60:.1f} 分钟；{a.iters} 步，LoRA r={a.rank}", flush=True)
    cfg = {
        "pretrained_path": MODEL.replace("\\", "/"), "train_manifest": manifest.replace("\\", "/"),
        "val_manifest": "", "sample_rate": 16000, "out_sample_rate": 48000,
        "batch_size": 1, "grad_accum_steps": a.accum, "num_workers": 0,
        "num_iters": a.iters, "log_interval": 10, "valid_interval": 10 ** 9, "save_interval": a.save_every or a.iters,
        "learning_rate": a.lr, "weight_decay": 0.01, "warmup_steps": min(100, a.iters // 10), "max_steps": a.iters,
        "max_batch_tokens": 0, "max_grad_norm": 1.0,
        "save_path": out.replace("\\", "/"), "tensorboard": os.path.join(out, "logs").replace("\\", "/"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": a.rank, "alpha": a.rank, "dropout": 0.0},
    }
    import yaml
    cfg_path = os.path.join(out, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return cfg_path


def patch_low_vram():
    import torch
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    orig = VoxCPM2Model.from_local.__func__

    def from_local(cls, *args, **kw):
        m = orig(cls, *args, **kw)
        if kw.get("training"):
            n = 0
            for name, p in m.named_parameters():
                if not p.requires_grad and "audio_vae" not in name and p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
                    n += p.numel()
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"[低显存] 冻结权重转 bf16：{n / 1e9:.2f}B 参数；可训练 LoRA 参数：{trainable / 1e6:.1f}M", flush=True)
        return m

    VoxCPM2Model.from_local = classmethod(from_local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=0)
    a = ap.parse_args()
    cfg = build(a.pid, a)
    patch_low_vram()
    sys.argv = [os.path.join(SRC, "scripts", "train_voxcpm_finetune.py"), "--config_path", cfg]
    runpy.run_path(sys.argv[0], run_name="__main__")
    steps = sorted(d for d in os.listdir(os.path.dirname(cfg)) if d.startswith("step_"))
    if steps:
        print("@@LORA@@" + os.path.join(os.path.dirname(cfg), steps[-1]), flush=True)


if __name__ == "__main__":
    main()
