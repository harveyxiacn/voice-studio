# -*- coding: utf-8 -*-
"""声音工坊命令行训练。关掉网页也不会中断，适合长时间训练。

交互菜单：  训练.bat
直接训练：  训练.bat train <声音名称或项目ID> --engine voxcpm|sovits|all [--iters 800]
查看项目：  训练.bat list
"""
import argparse
import os
import sys
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import voice_studio  # noqa: E402,F401  复用运行环境设置
import core  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S ") + str(msg), flush=True)


def show_projects():
    projs = core.list_projects()
    if not projs:
        print("还没有声音项目，请先在网页「① 上传录音」里上传并校对。")
        return []
    print("\n编号  声音名称            段数  已校对   有效时长   已有训练")
    for i, (name, pid) in enumerate(projs, 1):
        s = core.project_stats(core.load_project(pid))
        v = os.path.join(core.VOICES, pid, "voice.json")
        done = []
        if os.path.exists(v):
            vv = core.load_voice(pid)
            done = [t for t, ok in (("VoxCPM2", vv.get("voxcpm_lora")), ("SoVITS", vv["trained"])) if ok]
        print(f"{i:>3}   {name:<18}{s['kept']:>4}   {s['checked']:>3}/{s['kept']:<4} {s['sec'] / 60:>6.1f} 分钟   {'、'.join(done) or '-'}")
    return projs


def find(key):
    for name, pid in core.list_projects():
        if key in (name, pid):
            return core.load_project(pid)
    sys.exit(f"找不到声音项目：{key}")


def check_gpu():
    name, total, used = core.gpu_info()
    log(f"显卡：{name}，显存已用 {used}/{total} MB")
    if used > 1500:
        log("⚠️ 显存已被其他程序占用较多，请关闭剪映、微信、浏览器等软件，否则可能显存不足。")


def make_progress():
    state = {"t0": time.time(), "last": -1}

    def progress(pct, text):
        p = int(pct * 100)
        if p == state["last"]:
            return
        state["last"] = p
        el = time.time() - state["t0"]
        eta = f"，预计还需 {el * (1 - pct) / pct / 60:.0f} 分钟" if pct > 0.03 else ""
        bar = "█" * (p // 5) + "░" * (20 - p // 5)
        print(f"   [{bar}] {p:3d}%  {text}{eta}", flush=True)
    return progress


def _wait_msg():
    last = {"pos": None}

    def f(pos, cur):
        if last["pos"] != pos:
            last["pos"] = pos
            log(f"⏳ 显卡正忙（{cur}），排队中：第 {pos} 位，轮到后自动开始（Ctrl+C 可取消）")
    return f


def run(proj, engine, iters=None, e_sovits=8, e_gpt=15, bs=4):
    import gpu
    st = core.project_stats(proj)
    if st["sec"] < 10:
        sys.exit("有效人声不足 10 秒，无法训练")
    check_gpu()
    t0 = time.time()
    if not os.path.exists(os.path.join(core.VOICES, proj["id"], "voice.json")):
        core.build_voice(proj)
        log("已生成语音包（零样本）")
    try:
        if engine in ("voxcpm", "all"):
            with gpu.use("VoxCPM2 微调（命令行）", on_wait=_wait_msg()):
                gpu.check_free("VoxCPM2 微调")
                iters = iters or core.suggest_voxcpm_iters(st["sec"])
                lora = core.train_voxcpm(proj, iters, log, make_progress())
                core.build_voice(core.load_project(proj["id"]), voxcpm_lora=lora)
            log(f"✅ VoxCPM2 微调完成（累计 {(time.time() - t0) / 60:.1f} 分钟）")
        if engine in ("sovits", "all"):
            with gpu.use("GPT-SoVITS 训练（命令行）", on_wait=_wait_msg()):
                gpu.check_free("GPT-SoVITS 训练")
                sovits, gpt = core.train(core.load_project(proj["id"]), e_sovits, e_gpt, bs, log, make_progress())
                core.build_voice(core.load_project(proj["id"]), sovits, gpt)
            log(f"✅ GPT-SoVITS 训练完成（累计 {(time.time() - t0) / 60:.1f} 分钟）")
    except gpu.GpuBusy as e:
        sys.exit(f"❌ {e}\n（如果网页里加载了引擎，请在网页顶部点「🧹 释放显存」）")
    log(f"全部完成，语音包：{os.path.join(core.VOICES, proj['id'])}。打开声音工坊网页即可使用。")


def interactive():
    print("=" * 60 + "\n  声音工坊 · 命令行训练\n" + "=" * 60)
    projs = show_projects()
    if not projs:
        return
    sel = input("\n请输入要训练的项目编号：").strip()
    if not sel.isdigit() or not 1 <= int(sel) <= len(projs):
        sys.exit("编号无效")
    proj = core.load_project(projs[int(sel) - 1][1])
    st = core.project_stats(proj)
    print(f"\n已选：{proj['name']}（{st['sec'] / 60:.1f} 分钟）\n{core.duration_advice(st['sec'])}")
    if st["checked"] < st["kept"]:
        print(f"提示：还有 {st['kept'] - st['checked']} 段未校对，文稿错误会影响发音准确度。")
    print("\n选择引擎：\n  1. VoxCPM2 LoRA 微调（推荐，效果最好）\n  2. GPT-SoVITS 训练\n  3. 两个都训练")
    eng = {"1": "voxcpm", "2": "sovits", "3": "all"}.get(input("请输入 1/2/3（默认 1）：").strip() or "1")
    if not eng:
        sys.exit("选择无效")
    iters = None
    if eng in ("voxcpm", "all"):
        rec = core.suggest_voxcpm_iters(st["sec"])
        s = input(f"VoxCPM2 训练步数（直接回车用推荐值 {rec}）：").strip()
        iters = int(s) if s.isdigit() else rec
    input("\n请确认已关闭剪映等占显存的软件，按回车开始训练（训练中按 Ctrl+C 可停止）…")
    run(proj, eng, iters)


def main():
    if len(sys.argv) == 1:
        return interactive()
    ap = argparse.ArgumentParser(description="声音工坊命令行训练")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    t = sub.add_parser("train")
    t.add_argument("project")
    t.add_argument("--engine", choices=["voxcpm", "sovits", "all"], default="voxcpm")
    t.add_argument("--iters", type=int, default=None, help="VoxCPM2 训练步数（默认按素材时长推荐）")
    t.add_argument("--sovits_epochs", type=int, default=8)
    t.add_argument("--gpt_epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "list":
        show_projects()
    elif a.cmd == "train":
        run(find(a.project), a.engine, a.iters, a.sovits_epochs, a.gpt_epochs, a.batch)
    else:
        ap.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        core.stop_all()
        print("\n已停止。")
