# -*- coding: utf-8 -*-
"""显卡守卫：同一时间只允许一个显卡任务（网页和命令行共用文件锁），并在开始前检查显存是否够用。"""
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager

import core

LOCK_FILE = os.path.join(core.WORK, "_gpu.lock")
_local = threading.Lock()

# 常见占显存的软件（进程名小写 → 显示名）
HEAVY_APPS = {
    "jianyingpro.exe": "剪映", "capcut.exe": "CapCut", "adobe premiere pro.exe": "Premiere",
    "afterfx.exe": "After Effects", "resolve.exe": "达芬奇", "obs64.exe": "OBS",
    "photoshop.exe": "Photoshop", "blender.exe": "Blender", "potplayermini64.exe": "PotPlayer",
    "steam.exe": "Steam", "lghub.exe": None,
}
# 各类任务大约需要的空闲显存（MB，RTX 3070 8GB 实测 + 余量）
NEED_MB = {
    "VoxCPM2 微调": 7300, "GPT-SoVITS 训练": 4500, "VoxCPM2": 6300, "CosyVoice3": 4300,
    "GPT-SoVITS": 2000, "Qwen3-TTS": 5500, "识别处理": 2500, "人声分离": 3000,
}


class GpuBusy(RuntimeError):
    pass


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True


def holder():
    """当前占用显卡的任务 {"pid","task","since"}；进程已退出的旧锁自动清除。"""
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            h = json.load(f)
    except Exception:
        return None
    if not _pid_alive(h.get("pid", 0)):
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
        return None
    return h


def heavy_apps():
    try:
        import psutil
        names = {p.info["name"].lower() for p in psutil.process_iter(["name"]) if p.info.get("name")}
    except Exception:
        return []
    return sorted({v for k, v in HEAVY_APPS.items() if v and k in names})


def vram():
    """(已用 MB, 总 MB)"""
    _, total, used = core.gpu_info()
    return used, total


def check_free(task, already_loaded=False):
    """显存不够时抛出 GpuBusy，并说明是谁占着。"""
    need = 0 if already_loaded else NEED_MB.get(task, 0)
    if not need:
        return
    used, total = vram()
    if not total:
        return
    free = total - used
    if free >= need:
        return
    apps = heavy_apps()
    msg = f"显存不足：{task} 约需 {need / 1024:.1f}GB，当前只剩 {free / 1024:.1f}GB（已用 {used / 1024:.1f}/{total / 1024:.0f}GB）。"
    if apps:
        msg += f"检测到正在运行：{'、'.join(apps)}，请先关闭。"
    else:
        msg += "请关闭占用显卡的软件（浏览器视频、游戏、剪辑软件等）后重试。"
    raise GpuBusy(msg)


@contextmanager
def use(task):
    """独占显卡。网页内的并发请求、以及命令行训练都会互相排斥。"""
    h = holder()
    if h and h.get("pid") != os.getpid():
        mins = (time.time() - h.get("since", time.time())) / 60
        raise GpuBusy(f"显卡正忙：{h.get('task')}（另一个窗口，已运行 {mins:.0f} 分钟），请等它完成后再试。")
    if not _local.acquire(blocking=False):
        h = holder() or {}
        raise GpuBusy(f"显卡正忙：{h.get('task', '另一个任务')}，请等它完成后再试。")
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "task": task, "since": time.time()}, f, ensure_ascii=False)
        yield
    finally:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
        _local.release()


def status():
    used, total = vram()
    h = holder()
    return {"used": used, "total": total, "task": h.get("task") if h else None,
            "since": h.get("since") if h else None, "apps": heavy_apps()}
