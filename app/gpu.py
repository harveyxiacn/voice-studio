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


def _keep_awake(on):
    """任务期间阻止 Windows 睡眠（实测：笔记本睡眠会让训练停住数小时）。显示器仍可关闭。"""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


QUEUE_DIR = os.path.join(core.WORK, "_gpu_queue")
os.makedirs(QUEUE_DIR, exist_ok=True)


def queue():
    """排队中的任务（先来先到）：[{"id","pid","task","since"}]；进程已退出的自动清除。"""
    out = []
    for fn in sorted(os.listdir(QUEUE_DIR)):
        p = os.path.join(QUEUE_DIR, fn)
        try:
            with open(p, encoding="utf-8") as f:
                q = json.load(f)
        except Exception:
            continue
        if not _pid_alive(q.get("pid", 0)):
            try:
                os.remove(p)
            except OSError:
                pass
            continue
        q["id"] = fn
        out.append(q)
    return out


def cancel(qid=None):
    """取消排队：指定 id 或取消本进程的全部排队。"""
    for q in queue():
        if qid in (None, q["id"]) and (qid or q["pid"] == os.getpid()):
            open(os.path.join(QUEUE_DIR, q["id"] + ".cancel"), "w").close()


@contextmanager
def use(task, wait=True, on_wait=None):
    """独占显卡；显卡正忙时排队等待（先来先到，网页和命令行共用队列），轮到了自动开始。
    wait=False 时不排队、直接报「显卡正忙」。on_wait(位置, 正在运行的任务) 用于显示排队进度。任务期间阻止系统睡眠。"""
    def busy():
        h = holder()
        return h and h.get("pid") != os.getpid()

    if not wait and busy():
        raise GpuBusy(f"显卡正忙：{(holder() or {}).get('task')}，请等它完成后再试。")
    qid = f"{time.time():.3f}_{os.getpid()}_{threading.get_ident()}.json"
    qpath = os.path.join(QUEUE_DIR, qid)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "task": task, "since": time.time()}, f, ensure_ascii=False)
    try:
        while True:
            if os.path.exists(qpath + ".cancel"):
                raise GpuBusy("已取消排队")
            q = queue()
            pos = next((i for i, x in enumerate(q) if x["id"] == qid), 0) + 1
            if pos == 1 and not busy() and _local.acquire(blocking=False):
                break
            if on_wait:
                on_wait(pos, (holder() or {}).get("task") or (q[0]["task"] if q else ""))
            time.sleep(2)
    finally:
        for p in (qpath, qpath + ".cancel"):
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "task": task, "since": time.time()}, f, ensure_ascii=False)
        _keep_awake(True)
        yield
    finally:
        _keep_awake(False)
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
        _local.release()


def status():
    used, total = vram()
    h = holder()
    return {"used": used, "total": total, "task": h.get("task") if h else None,
            "since": h.get("since") if h else None, "apps": heavy_apps(), "queue": [q["task"] for q in queue()]}
