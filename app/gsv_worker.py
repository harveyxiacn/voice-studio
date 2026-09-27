# -*- coding: utf-8 -*-
"""GPT-SoVITS 推理工作进程（用 GPT-SoVITS 自带 runtime 运行），协议同 engines/voxcpm_worker.py。
放在独立进程里，退出后显存完全释放，不会影响训练。"""
import json
import os
import sys
import time
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import core  # noqa: E402

os.chdir(core.GSV)
for p in (core.GSV, os.path.join(core.GSV, "GPT_SoVITS"), os.path.join(core.GSV, "tools")):
    sys.path.insert(0, p)
os.environ["version"] = "v2"


def reply(obj):
    sys.stdout.write("@@RESULT@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    from tts_engine import ENGINE
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            cmd = json.loads(line)
            t0 = time.time()
            if cmd["cmd"] == "quit":
                reply({"ok": True})
                break
            if cmd["cmd"] == "load":
                res = {"ok": True}
            elif cmd["cmd"] == "gsv":
                out, seed, rep = ENGINE.synthesize(*cmd["args"])
                if cmd.get("out"):
                    import shutil
                    shutil.move(out, cmd["out"])
                    out = cmd["out"]
                import torch
                res = {"ok": True, "out": out, "seed": seed, "report": rep,
                       "vram": int(torch.cuda.max_memory_allocated() / 2 ** 20)}
            else:
                res = {"ok": False, "error": "unknown cmd"}
            res["sec"] = round(time.time() - t0, 1)
            reply(res)
        except Exception as e:
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
