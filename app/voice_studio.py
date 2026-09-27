# -*- coding: utf-8 -*-
"""声音工坊 WebUI：导入素材（自动分人/人声分离）→ 录音向导 → 逐段校对 → 语音包与训练 → 文字转语音（字幕/单句重做/历史）→ 管理。"""
import os
import sys
import threading
import time
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import core  # noqa: E402

# 运行环境：以 GPT-SoVITS 目录为工作目录，并让子进程能找到它的模块
os.chdir(core.GSV)
for p in (core.GSV, os.path.join(core.GSV, "GPT_SoVITS"), os.path.join(core.GSV, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PATH"] = os.path.join(core.GSV, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["version"] = "v2"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
_site = os.path.join(core.GSV, "runtime", "Lib", "site-packages")
g = core.GSV.replace("\\", "/")
with open(os.path.join(_site, "users.pth"), "w") as f:
    f.write(f"{g}\n{g}/GPT_SoVITS/BigVGAN\n{g}/tools\n{g}/tools/asr\n{g}/GPT_SoVITS\n{g}/tools/uvr5")

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda: None
except Exception:
    pass
import gradio as gr  # noqa: E402

import engines as ENG  # noqa: E402
import gpu  # noqa: E402
import recorder  # noqa: E402
import takes  # noqa: E402
from tts_engine import TEXT_LANGS  # noqa: E402

PAGE = 8
MAX_SPK = 8
LANG_CHOICES = list(core.LANGS.keys())
PREVIEW_DIR = os.path.join(core.WORK, "_preview")
SAMPLE = {"zh": "大家好，这是用我的声音做出来的语音包，你听听看，像不像我本人？",
          "yue": "大家好，呢个系用我把声整出嚟嘅语音包，你听下似唔似我本人？"}


# ------------------------------------------------------------------ 通用：后台任务 + 进度
def progress_html(prog, done=False):
    if prog.get("pct") is None:
        return ""
    pct = 1.0 if done and prog.get("ok") else prog["pct"]
    el = time.time() - prog["t0"]
    eta = ""
    if not done and 0.03 < pct < 1:
        rest = el * (1 - pct) / pct
        eta = f"，预计还需 {rest / 60:.0f} 分钟" if rest >= 90 else f"，预计还需 {rest:.0f} 秒"
    color = "#16a34a" if done and prog.get("ok") else ("#dc2626" if done else "#2563eb")
    text = "完成" if done and prog.get("ok") else ("已停止 / 出错" if done else prog.get("text", ""))
    return (f"<div class='pbar'><div class='pfill' style='width:{pct * 100:.1f}%;background:{color}'></div></div>"
            f"<div class='ptext'>{pct * 100:.0f}%　{text}　（已用 {el / 60:.1f} 分钟{eta}）</div>")


def stream_task(fn, *args):
    """在线程里运行耗时任务，边跑边推送日志和进度。fn(log, *args) 返回最终提示；log.progress(pct, text) 更新进度。
    每次 yield (日志, 最终提示或 None, 进度 HTML)。"""
    logs, result, prog = [], {}, {"pct": None, "text": "", "t0": time.time()}

    def log(msg):
        if msg:
            logs.append(time.strftime("%H:%M:%S ") + str(msg))

    log.progress = lambda pct, text="": prog.update(pct=max(0.0, min(1.0, pct)), text=text)

    def run():
        try:
            result["msg"] = fn(log, *args)
            prog["ok"] = True
        except Exception as e:
            if not isinstance(e, (gpu.GpuBusy, ValueError)):
                traceback.print_exc()
            result["msg"] = f"❌ {e}"
            log(f"❌ {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while t.is_alive():
        time.sleep(1)
        yield "\n".join(logs[-300:]), None, progress_html(prog)
    yield "\n".join(logs[-300:]), result.get("msg", ""), progress_html(prog, done=True)


def _queued(log, task):
    """显卡正忙时排队（先来先到），在日志和进度条上显示排队位置，轮到了自动开始。"""
    said = {"pos": None}

    def on_wait(pos, cur):
        if said["pos"] != pos:
            said["pos"] = pos
            log(f"⏳ 显卡正忙（{cur}），排队中：第 {pos} 位，轮到后自动开始")
        log.progress(0.0, f"排队中：第 {pos} 位（等待「{cur}」完成）")

    return gpu.use(task, on_wait=on_wait)


def project_choices():
    return [(n, pid) for n, pid in core.list_projects()]


def _valid(pid):
    """项目可能已在「管理」里删除或合并：不存在时当作未选择。"""
    return pid if pid and os.path.exists(os.path.join(core.project_dir(pid), "project.json")) else None


def voice_choices():
    return core.list_voices()


def stats_md(pid):
    pid = _valid(pid)
    if not pid:
        return "尚未选择声音。请在页面顶部选择或新建一个声音。"
    proj = core.load_project(pid)
    s = core.project_stats(proj)
    return (f"**{proj['name']}**：共 {s['n']} 段，保留 {s['kept']} 段，已校对 **{s['checked']}/{s['kept']}**，"
            f"有效人声 **{s['sec']/60:.1f} 分钟**\n\n{core.duration_advice(s['sec'])}")


# ------------------------------------------------------------------ 顶部：状态栏 + 步骤清单
def status_html():
    st = gpu.status()
    used, total = st["used"], st["total"]
    pct = used / total * 100 if total else 0
    col = "#16a34a" if pct < 50 else ("#d97706" if pct < 85 else "#dc2626")
    chips = [f"<span class='chip'>🖥️ 显存 <b style='color:{col}'>{used / 1024:.1f}</b>/{total / 1024:.0f} GB</span>"]
    ld = ENG.loaded()
    chips.append(f"<span class='chip'>🔊 引擎：{'、'.join(ld) + ' 已加载' if ld else '未加载'}</span>")
    if st["task"]:
        mins = (time.time() - (st["since"] or time.time())) / 60
        chips.append(f"<span class='chip busy'>⏳ {st['task']}（{mins:.0f} 分钟）</span>")
    else:
        chips.append("<span class='chip'>✅ 空闲</span>")
    if st["queue"]:
        chips.append(f"<span class='chip busy'>🕒 排队 {len(st['queue'])} 个：{'、'.join(st['queue'][:3])}</span>")
    if st["apps"]:
        chips.append(f"<span class='chip warn'>⚠️ 检测到 {'、'.join(st['apps'])} 正在运行，训练前请关闭</span>")
    return "<div class='statusbar'>" + "".join(chips) + "</div>"


def on_release():
    if gpu.holder():
        gr.Warning("有任务正在使用显卡，暂时不能释放")
    else:
        ENG.release_all()
        gr.Info("已释放所有引擎占用的显存")
    return status_html()


def checklist_html(pid):
    pid = _valid(pid)
    if not pid:
        return ("<div class='steps'><span class='step now'>👉 第一步：在上方新建一个声音（写个名字点「打开 / 新建」），"
                "或者直接到「① 导入素材」上传视频，系统会自动分人建好声音</span></div>")
    proj = core.load_project(pid)
    s = core.project_stats(proj)
    has_voice = os.path.exists(os.path.join(core.VOICES, pid, "voice.json"))
    v = core.load_voice(pid) if has_voice else None
    tuned = bool(v and (v.get("voxcpm_lora") or v["trained"]))
    made = any(m[1] for m in takes.history(200) if takes.load(m[1])["vid"] == pid) if has_voice else False
    steps = [
        (s["sec"] > 0, f"① 素材 {s['sec'] / 60:.1f} 分钟", "到「① 导入素材」上传视频，或到「🎤 录音向导」照着台词录"),
        (s["kept"] > 0 and s["checked"] >= s["kept"], f"② 校对 {s['checked']}/{s['kept']}", "到「② 逐段校对」核对文字（先看 ⚠️ 疑点）"),
        (has_voice, "③ 语音包", "到「③ 语音包与训练」点「生成 / 更新语音包」"),
        (tuned, "④ 微调（可选）", "素材够 5 分钟后，到「③」做 VoxCPM2 微调，会更像本人"),
        (made, "⑤ 生成语音", "到「④ 文字转语音」输入文案生成，带字幕"),
    ]
    html, nxt = [], None
    for ok, label, hint in steps:
        cls = "done" if ok else ("now" if nxt is None else "")
        if not ok and nxt is None:
            nxt = hint
        html.append(f"<span class='step {cls}'>{'✅' if ok else '⬜'} {label}</span>")
    if s["sec"] < core.RECOMMEND_MIN_SEC and s["sec"] > 0:
        nxt = (nxt or "") + f"　｜　素材只有 {s['sec']:.0f} 秒，建议补充到 1 分钟以上（录音向导最快）"
    return (f"<div class='steps'><b>{proj['name']}</b>　" + " → ".join(html) +
            (f"<div class='next'>👉 下一步：{nxt}</div>" if nxt else "<div class='next'>🎉 全部完成！</div>") + "</div>")


def on_open_project(name, sel):
    try:
        if name and name.strip():
            proj = core.create_project(name)
        elif sel:
            proj = core.load_project(sel)
        else:
            raise ValueError("请填写新声音名称，或从下拉框选择已有声音")
    except Exception as e:
        gr.Warning(str(e))
        return gr.update(), gr.update(), ""
    return gr.update(choices=project_choices(), value=proj["id"]), proj["id"], ""


# ------------------------------------------------------------------ ① 导入素材
def probe_duration(files):
    import subprocess
    total = 0.0
    for f in files or []:
        path = f if isinstance(f, str) else f.name
        try:
            out = subprocess.check_output(
                [os.path.join(core.GSV, "runtime", "ffprobe.exe"), "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", path],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            total += float(out.decode().strip())
        except Exception:
            pass
    return total


def on_files_change(files):
    if not files:
        return "上传后这里会显示总时长和建议。"
    sec = probe_duration(files)
    return (f"已选 {len(files)} 个文件，总时长约 **{sec / 60:.1f} 分钟**（含静音）。\n\n"
            "每个人建议最短 **1 分钟**（最少 10 秒），**5–30 分钟**效果最好，**不设上限**。")


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in files or []]


def _maybe_separate(log, paths, sep, dereverb):
    if not sep:
        return None
    log(f"🎵 人声分离中（BS-Roformer{' + 去混响' if dereverb else ''}，去掉背景音乐和噪音）…")
    log.progress(0.05, "人声分离")
    return core.separate_files(paths, log, dereverb)


def do_direct(log, pid, files, lang, sep, dereverb):
    if not pid:
        raise ValueError("请先在页面顶部选择或新建一个声音")
    if not files:
        raise ValueError("请先上传录音或视频文件")
    with _queued(log, "导入素材"):
        ENG.release_all()
        gpu.check_free("识别处理")
        paths = _paths(files)
        clean = _maybe_separate(log, paths, sep, dereverb) or paths
        log.progress(0.4, "切片与识别")
        total, n = core.add_audio(core.load_project(pid), clean, lang, log)
    s = core.project_stats(core.load_project(pid))
    log(f"✅ 完成：本次 {total / 60:.1f} 分钟录音切成 {n} 段。")
    return f"✅ 导入完成，共 {n} 段。请到「② 逐段校对」检查文稿。\n\n" + core.duration_advice(s["sec"])


def on_direct(pid, files, lang, sep, dereverb):
    for logs, msg, _ in stream_task(do_direct, pid, files, lang, sep, dereverb):
        yield logs, (msg if msg is not None else gr.update())


def on_analyze(files, lang, sep, dereverb, sens, num):
    """（可选人声分离）→ 识别说话人 → 自动认人 → 生成试听 → 填充确认面板。"""
    holder = {}

    def job(log):
        if not files:
            raise ValueError("没有收到文件：请先上传录音或视频。大文件上传需要一点时间，等文件名旁边显示出大小后再点。")
        with _queued(log, "识别说话人"):
            ENG.release_all()
            gpu.check_free("识别处理")
            paths = _paths(files)
            clean = _maybe_separate(log, paths, sep, dereverb)
            log.progress(0.5, "识别说话人")
            log("🔍 识别说话人中…")
            # 用原始音频识别说话人（人声分离会抹平音色差异），从分离后的干净人声里截取
            res = core.diarize_files(paths, log, threshold=float(sens), num=int(num or 0))
            if clean:
                for f, c in zip(res["files"], clean):
                    f["audio"] = c
            if not res["speakers"]:
                raise ValueError("没有检测到人声")
            log.progress(0.9, "生成试听片段")
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            for sp in res["speakers"][:MAX_SPK]:
                sp["preview"] = os.path.join(PREVIEW_DIR, f"spk{sp['id'] + 1}_{stamp}.wav")
                core.speaker_audio(res, {sp["id"]}, sp["preview"], max_sec=12)
        res["match"] = core.match_voiceprints(res)
        holder["res"] = res
        n = len(res["speakers"])
        known = sum(1 for m in res["match"] if m)
        return (f"🔍 识别到 **{n} 位说话人**" + (f"，其中 {known} 位认出是已有的声音（已自动填好名字）" if known else "") +
                "。请试听每个人的片段，确认名字后点「✅ 确认并生成语音包」。" +
                ("\n\n⚠️ 超过 8 人，只显示说话时间最多的 8 位。" if n > MAX_SPK else ""))

    blank = [gr.update()] * (MAX_SPK * 6)
    logs = msg = ""
    for logs, msg, prog in stream_task(job):
        if msg is None:
            yield [logs, gr.update(), gr.update(), gr.update()] + blank
    res = holder.get("res")
    if not res:
        yield [logs, msg, None, gr.update(visible=False)] + blank
        return
    hints = core.speaker_hints(res)
    names = [n for n, _ in core.list_projects()]
    outs = []
    for k in range(MAX_SPK):
        if k < len(res["speakers"]):
            sp = res["speakers"][k]
            m = res["match"][k] if k < len(res["match"]) else None
            h = ([f"🧠 很可能是已有的「{m[0]}」（声纹相似度 {m[1]:.2f}），确认后会追加进去"] if m else []) + ([hints[k]] if hints[k] else [])
            title = f"**说话人{k + 1}**　说话 {sp['dur']:.0f} 秒 · {sp['n_turns']} 段" + ("\n\n" + "\n\n".join(h) if h else "")
            outs += [gr.update(visible=True), title, sp["preview"],
                     gr.update(value=m[0] if m else f"说话人{k + 1}", choices=names), lang, sp["dur"] >= 10]
        else:
            outs += [gr.update(visible=False), "", None, gr.update(value=""), lang, False]
    yield [logs, msg, res, gr.update(visible=True)] + outs


def on_confirm_speakers(res, *vals):
    if not res:
        raise gr.Error("请先识别说话人")
    names, langs_, keeps_ = vals[:MAX_SPK], vals[MAX_SPK:2 * MAX_SPK], vals[2 * MAX_SPK:]
    assign = [{"spk": k, "name": names[k] or "", "lang": langs_[k], "keep": bool(keeps_[k])}
              for k in range(min(MAX_SPK, len(res["speakers"])))]
    holder = {}

    def job(log):
        with _queued(log, "导入素材"):
            ENG.release_all()
            holder["made"] = core.process_speakers(res, assign, log)
        names_done = [core.load_project(p)["name"] for p in holder["made"]]
        log("✅ 全部完成：" + "、".join(names_done))
        return (f"✅ 已处理 {len(names_done)} 个声音：**{'、'.join(names_done)}**。"
                "都已生成语音包，可以直接到「④ 文字转语音」试听；到「② 逐段校对」修正文稿后效果更好。")

    logs = msg = ""
    for logs, msg, _ in stream_task(job):
        if msg is None:
            yield logs, gr.update(), gr.update(), gr.update(), gr.update(visible=True)
    made = holder.get("made") or []
    first = made[0] if made else None
    yield (logs, msg, gr.update(choices=project_choices(), value=first) if first else gr.update(),
           first if first else gr.update(), gr.update(visible=not made))


# ------------------------------------------------------------------ 🎤 录音向导
EMO_COLOR = {"平静": "#64748b", "开心": "#f59e0b", "兴奋": "#ef4444", "温柔": "#ec4899",
             "悲伤": "#3b82f6", "生气": "#b91c1c", "惊讶": "#8b5cf6", "严肃": "#0f766e"}


def rec_view(pid, lang, emo_filter, idx):
    """返回：台词卡 HTML、进度 HTML、规范化后的序号。"""
    pid = _valid(pid)
    lst = recorder.script_list(lang, emo_filter)
    idx = int(idx or 0) % max(len(lst), 1)
    rec = core.load_project(pid).get("recorded", {}) if pid else {}
    sid, emo, text = lst[idx]
    done = sid in rec
    card = (f"<div class='card'><div class='card-top'><span class='emo' style='background:{EMO_COLOR[emo]}'>{emo}</span>"
            f"<span class='cnt'>第 {idx + 1}/{len(lst)} 句{'　✅ 已录（可重录覆盖）' if done else ''}</span></div>"
            f"<div class='line'>{text}</div><div class='tip'>💡 {recorder.TIPS[emo]}</div></div>")
    d, t, per = recorder.progress(pid, lang)
    chips = "".join(f"<span class='chip' style='border-color:{EMO_COLOR[e]}'>{e} {a}/{b}</span>" for e, (a, b) in per.items())
    prog = (f"<div class='recprog'><b>{lang}台词已录 {d}/{t}</b>（约 {d * 5 / 60:.1f} 分钟）　{chips}</div>"
            if pid else "<div class='recprog'>⚠️ 请先在页面顶部选择或新建一个声音，再开始录音</div>")
    return card, prog, idx


def rec_move(pid, lang, emo_filter, idx, delta):
    return rec_view(pid, lang, emo_filter, int(idx or 0) + delta) + ("", "")


def _save_b64(data_url):
    """浏览器录音（data URL，一般是 webm/opus）→ 临时文件，交给 ffmpeg 解码。"""
    import base64
    head, _, body = data_url.partition(",")
    ext = ".ogg" if "ogg" in head else (".wav" if "wav" in head else ".webm")
    path = os.path.join(core.WORK, f"_rec_{time.strftime('%H%M%S')}{ext}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(body))
    return path


def rec_save(pid, lang, emo_filter, idx, b64, custom, custom_emo):
    if not pid:
        raise gr.Error("请先在页面顶部选择或新建一个声音")
    if not b64 or "," not in b64:
        raise gr.Error("还没有录音：点「🎙️ 开始录音」，念完点「⏹ 停止」")
    if custom and custom.strip():
        text, emo, sid = custom.strip(), custom_emo or "平静", ""
    else:
        sid, emo, text = recorder.script_list(lang, emo_filter)[int(idx or 0)]
    path = _save_b64(b64)
    try:
        ok, msg = recorder.save_recording(pid, path, text, emo, lang, sid)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not ok:
        card, prog, idx = rec_view(pid, lang, emo_filter, idx)
        return card, prog, idx, gr.update(), msg, gr.update()
    card, prog, idx = rec_view(pid, lang, emo_filter, int(idx) + (0 if custom and custom.strip() else 1))
    return card, prog, idx, "", f"{msg}　→ 已保存：「{text[:24]}」", gr.update(value="")


# 自制录音器：只在点「开始录音」时申请麦克风，点「停止」后立即释放（浏览器的麦克风图标会消失）
RECORDER_HTML = """
<div class='recorder'>
  <button id='vs-start' class='rbtn start' onclick='vsRecStart()'>🎙️ 开始录音</button>
  <button id='vs-stop' class='rbtn stop' onclick='vsRecStop()' disabled>⏹ 停止</button>
  <span id='vs-status' class='rstatus'>点「开始录音」时才会使用麦克风，停止后立即关闭</span>
  <audio id='vs-preview' controls style='display:none; width:100%; margin-top:8px'></audio>
</div>
"""

RECORDER_JS = """
<script>
let vsRec = null, vsStream = null, vsChunks = [], vsT0 = 0, vsTimer = null;
function vsStatus(t) { const s = document.getElementById('vs-status'); if (s) s.textContent = t; }
function vsSetB64(v) {
  const ta = document.querySelector('#rec-b64 textarea');
  if (!ta) return;
  ta.value = v;
  ta.dispatchEvent(new Event('input', { bubbles: true }));
}
async function vsRecStart() {
  if (vsRec) return;
  try {
    vsStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { vsStatus('❌ 无法使用麦克风：' + e.message + '（请在浏览器地址栏允许麦克风）'); return; }
  vsChunks = [];
  vsRec = new MediaRecorder(vsStream);
  vsRec.ondataavailable = (e) => { if (e.data.size) vsChunks.push(e.data); };
  vsRec.onstop = () => {
    vsStream.getTracks().forEach((t) => t.stop());   // 立即释放麦克风
    vsStream = null; vsRec = null;
    const blob = new Blob(vsChunks, { type: (vsChunks[0] && vsChunks[0].type) || 'audio/webm' });
    const a = document.getElementById('vs-preview');
    a.src = URL.createObjectURL(blob); a.style.display = 'block';
    const fr = new FileReader();
    fr.onload = () => vsSetB64(fr.result);
    fr.readAsDataURL(blob);
    vsStatus('✅ 录好了 ' + ((Date.now() - vsT0) / 1000).toFixed(1) + ' 秒，麦克风已关闭。可以先回放，满意就点「保存并下一句」');
  };
  vsRec.start();
  vsT0 = Date.now();
  document.getElementById('vs-start').disabled = true;
  document.getElementById('vs-stop').disabled = false;
  vsTimer = setInterval(() => vsStatus('🔴 录音中… ' + ((Date.now() - vsT0) / 1000).toFixed(1) + ' 秒（念完点停止）'), 200);
}
function vsRecStop() {
  if (!vsRec) return;
  clearInterval(vsTimer);
  vsRec.stop();
  document.getElementById('vs-start').disabled = false;
  document.getElementById('vs-stop').disabled = true;
}
function vsRecReset() {
  const a = document.getElementById('vs-preview');
  if (a) { a.removeAttribute('src'); a.style.display = 'none'; }
  vsStatus('点「开始录音」时才会使用麦克风，停止后立即关闭');
}
</script>
"""


# ------------------------------------------------------------------ ② 逐段校对
FILTERS = ["⚠️ 疑点优先", "全部", "未校对", "保留的", "已删除"]
NO_MOVE = "（不移动）"


def _filtered(proj, flt):
    segs = proj["segments"]
    idx = list(range(len(segs)))
    if flt == "未校对":
        idx = [i for i in idx if segs[i]["keep"] and not segs[i].get("checked")]
    elif flt == "已删除":
        idx = [i for i in idx if not segs[i]["keep"]]
    elif flt == "保留的":
        idx = [i for i in idx if segs[i]["keep"]]
    elif flt == "⚠️ 疑点优先":  # 未校对的按识别可信度从低到高
        idx = sorted([i for i in idx if segs[i]["keep"] and not segs[i].get("checked")],
                     key=lambda i: segs[i].get("conf", 0.5))
    return idx


def _conf_badge(s):
    if s.get("source") == "录音向导":
        return "🎤 录音向导"
    c = s.get("conf")
    if c is None:
        return ""
    if c >= 0.9:
        return f"🟢 可信 {c:.2f}"
    if c >= 0.7:
        return f"🟡 请留意 {c:.2f}"
    return f"🔴 疑点 {c:.2f}"


def render_page(pid, page, flt):
    """返回：页码, 页状态, 页信息, 以及每个槽位的 (分组, 标题, 音频, 文本, 语言, 情感, 保留, 移到)。"""
    pid = _valid(pid)
    outs = []
    others = [NO_MOVE] + [n for n, p in core.list_projects() if p != pid]
    if not pid:
        page_idx, total_pages, info = [], 1, "请先在页面顶部选择一个声音。"
    else:
        proj = core.load_project(pid)
        idx = _filtered(proj, flt)
        total_pages = max(1, (len(idx) + PAGE - 1) // PAGE)
        page = int(min(max(1, page or 1), total_pages))
        page_idx = idx[(page - 1) * PAGE: page * PAGE]
        info = f"第 {page}/{total_pages} 页（{flt}，共 {len(idx)} 段）"
        if flt == "⚠️ 疑点优先" and not idx:
            info = "🎉 没有待校对的段落了"
    for k in range(PAGE):
        if k < len(page_idx):
            i = page_idx[k]
            s = proj["segments"][i]
            alt = f"\n\n另一个识别结果：{s['alt']}" if s.get("alt") else ""
            flag = "✅ 已校对" if s.get("checked") else "⬜ 未校对"
            title = f"**#{i + 1}**　{s['dur']:.1f} 秒　{flag}　{_conf_badge(s)}　`Alt+{k + 1}` 播放{alt}"
            outs += [gr.update(visible=True), title, os.path.join(core.project_dir(pid), "slices", s["file"]),
                     s["text"], core.LANG_NAMES.get(s["lang"], "普通话"), s["emotion"], s["keep"],
                     gr.update(choices=others, value=NO_MOVE)]
        else:
            outs += [gr.update(visible=False), "", None, "", "普通话", "平静", True, gr.update(choices=others, value=NO_MOVE)]
    return [page if pid else 1, page_idx, info] + outs


def save_page(pid, page_idx, vals):
    if not pid or not page_idx:
        return
    proj = core.load_project(pid)
    P = PAGE
    texts, langs, emos, keeps, moves = vals[0:P], vals[P:2 * P], vals[2 * P:3 * P], vals[3 * P:4 * P], vals[4 * P:5 * P]
    to_move = {}
    for k, i in enumerate(page_idx):
        s = proj["segments"][i]
        s["text"] = (texts[k] or "").strip()
        s["lang"] = core.LANGS.get(langs[k], s["lang"])
        s["emotion"] = emos[k] or "平静"
        s["keep"] = bool(keeps[k]) and bool(s["text"])
        s["checked"] = True
        if moves[k] and moves[k] != NO_MOVE:
            to_move.setdefault(moves[k], []).append(i)
    core.save_project(proj)
    name2pid = {n: p for n, p in core.list_projects()}
    for name, idxs in to_move.items():
        if name in name2pid:
            n = core.move_segments(pid, idxs, name2pid[name])
            gr.Info(f"已把 {n} 段移到「{name}」")


def make_nav(delta):
    def f(pid, page, flt, page_idx, *vals):
        save_page(pid, page_idx, vals)
        # 疑点/未校对筛选下，保存后当前页的段落会离开列表，停留在原页即可看到下一批
        step = 0 if flt in ("⚠️ 疑点优先", "未校对") and delta > 0 else delta
        return render_page(pid, (page or 1) + step, flt) + [stats_md(pid)]
    return f


def bulk_emotion(page_idx, emo):
    return [gr.update(value=emo) if k < len(page_idx or []) else gr.update() for k in range(PAGE)]


# ------------------------------------------------------------------ ③ 语音包与训练
def default_batch():
    _, total, _ = core.gpu_info()
    return 4 if total >= 7000 else 2


def train_info(pid):
    pid = _valid(pid)
    if not pid:
        return "尚未选择声音。"
    proj = core.load_project(pid)
    s = core.project_stats(proj)
    vj = os.path.join(core.VOICES, pid, "voice.json")
    v = core.load_voice(pid) if os.path.exists(vj) else None
    lines = [stats_md(pid)]
    if v:
        emo = {}
        for r in v["refs"]:
            emo[r["emotion"]] = emo.get(r["emotion"], 0) + 1
        lines.append(f"**语音包**：已生成（{v.get('updated', v['created'])}），情感参考录音："
                     + "、".join(f"{k}×{n}" for k, n in emo.items())
                     + f"　｜　VoxCPM2 微调：{'✅' if v.get('voxcpm_lora') else '—'}　｜　GPT-SoVITS 训练：{'✅' if v['trained'] else '—'}")
        missing = [e for e in core.EMOTIONS if e not in emo]
        if missing:
            lines.append(f"缺少这些情感的参考录音：{'、'.join(missing)}（可以用「🎤 录音向导」补录）")
    else:
        lines.append("**语音包**：还没有生成，先点下面的「⚡ 生成 / 更新语音包」。")
    if s["sec"] < 300:
        lines.append(f"💡 VoxCPM2 微调建议素材 ≥5 分钟（现在 {s['sec'] / 60:.1f} 分钟）。素材太少时微调提升有限，零样本已经不错。")
    return "\n\n".join(lines)


def do_zero_shot(log, pid):
    if not pid:
        raise ValueError("请先选择一个声音")
    meta = core.build_voice(core.load_project(pid))
    log(f"✅ 语音包已更新：{meta['name']}（参考录音 {len(meta['refs'])} 条）")
    with _queued(log, "更新声纹"):
        core.update_voiceprint(pid, log)
    return "✅ 语音包已生成 / 更新，三个引擎都可以直接用了。到「④ 文字转语音」试听。"


def _pre_train(log, pid):
    if not pid:
        raise ValueError("请先选择一个声音")
    proj = core.load_project(pid)
    st = core.project_stats(proj)
    if st["sec"] < 10:
        raise ValueError("有效人声不足 10 秒，无法训练")
    if st["checked"] < st["kept"]:
        log(f"提示：还有 {st['kept'] - st['checked']} 段未校对，识别错误的文稿会影响发音准确度。")
    if not os.path.exists(os.path.join(core.VOICES, pid, "voice.json")):
        core.build_voice(proj)
        log("已先生成语音包")
    return proj


def _sample_take(engine, pid, use_lora, log):
    v = core.load_voice(pid)
    lang = "yue" if v["langs"] == ["yue"] else "zh"
    p = dict(lang_label="粤语（可夹英文）" if lang == "yue" else "普通话（可夹英文）", default_emo="平静", auto_emo=False,
             expr=0.6, speed=1.0, pause=0.3, seed=7, multi=True, strong=False, use_lora=use_lora)
    return takes.create(engine, pid, SAMPLE[lang], p, log)["wav"]


def do_train_vox(log, pid, iters, holder):
    proj = _pre_train(log, pid)
    with _queued(log, "VoxCPM2 微调"):
        ENG.release_all()
        gpu.check_free("VoxCPM2 微调")
        t0 = time.time()
        lora = core.train_voxcpm(proj, int(iters), log, log.progress)
        core.build_voice(core.load_project(pid), voxcpm_lora=lora)
        log(f"✅ 微调完成，用时 {(time.time() - t0) / 60:.1f} 分钟。生成训练前 / 后对比试听…")
        log.progress(0.99, "生成对比试听")
        holder["before"] = _sample_take("VoxCPM2", pid, False, log)
        holder["after"] = _sample_take("VoxCPM2", pid, True, log)
        ENG.release_all()
    return "✅ VoxCPM2 微调完成！下面可以对比「训练前 / 训练后」，之后用 VoxCPM2 生成会自动用上微调成果。"


def do_train_gsv(log, pid, e_sovits, e_gpt, bs, holder):
    proj = _pre_train(log, pid)
    with _queued(log, "GPT-SoVITS 训练"):
        ENG.release_all()
        gpu.check_free("GPT-SoVITS 训练")
        t0 = time.time()
        holder["before"] = _sample_take("GPT-SoVITS", pid, False, log) if not core.load_voice(pid)["trained"] else None
        ENG.release_all()
        sovits, gpt = core.train(proj, int(e_sovits), int(e_gpt), int(bs), log, log.progress)
        core.build_voice(core.load_project(pid), sovits, gpt)
        log(f"✅ 训练完成，用时 {(time.time() - t0) / 60:.1f} 分钟。生成试听…")
        holder["after"] = _sample_take("GPT-SoVITS", pid, True, log)
        ENG.release_all()
    return "✅ GPT-SoVITS 训练完成！下面可以试听训练后的效果。"


def _train_stream(fn, *args):
    holder = {}
    logs = msg = prog = ""
    for logs, msg, prog in stream_task(fn, *args, holder):
        if msg is None:
            yield logs, gr.update(), prog, gr.update(), gr.update(), gr.update()
    has = bool(holder.get("after"))
    yield (logs, msg, prog, gr.update(visible=has), holder.get("before"), holder.get("after"))


def on_train_vox(pid, iters):
    yield from _train_stream(do_train_vox, pid, iters)


def on_train_gsv(pid, a, b, c):
    yield from _train_stream(do_train_gsv, pid, a, b, c)


def on_zero(pid):
    for logs, msg, _ in stream_task(do_zero_shot, pid):
        yield logs, (msg if msg is not None else gr.update())


def on_stop():
    core.stop_all()
    return "⏹ 已发送停止指令。"


def suggest_iters(pid):
    pid = _valid(pid)
    if not pid:
        return gr.update()
    return gr.update(value=core.suggest_voxcpm_iters(core.project_stats(core.load_project(pid))["sec"]))


# ------------------------------------------------------------------ ④ 文字转语音（作品）
def voice_info(vid):
    if not vid:
        return "请选择语音包。"
    v = core.load_voice(vid)
    cnt = {}
    for r in v["refs"]:
        key = f"{r['emotion']}({core.LANG_NAMES.get(r['lang'], r['lang'])})"
        cnt[key] = cnt.get(key, 0) + 1
    st = [f"VoxCPM2：{'✅ 已微调' if v.get('voxcpm_lora') else '零样本'}", "CosyVoice3：零样本",
          f"GPT-SoVITS：{'✅ 已训练' if v['trained'] else '零样本（粤语易漏字，建议先训练）'}"]
    pins = v.get("pinned") or {}
    pin_txt = ("\n\n📌 已锁定参考录音：" + "、".join(f"{core.LANG_NAMES.get(k.split('|')[0], k)}·{k.split('|')[1]}" for k in pins)
               if pins else "")
    return (f"**{v['name']}**　" + "　|　".join(st) + "\n\n情感参考录音：" + "、".join(f"{k}×{n}" for k, n in cnt.items())
            + pin_txt)


def _params(tlang, emo, auto, expr, speed, pause, seed, multi, strong, use_lora):
    return dict(lang_label=tlang, default_emo=emo, auto_emo=bool(auto), expr=float(expr), speed=float(speed),
                pause=float(pause), seed=int(seed) if seed is not None else -1, multi=bool(multi),
                strong=bool(strong), use_lora=bool(use_lora))


def _take_outputs(meta, note=""):
    """作品 → (整段音频, 字幕文件, 说明, 句子表, 历史下拉)"""
    miss = [i + 1 for i, s in enumerate(meta["sentences"]) if not s.get("hit", True)]
    md = (f"**{meta['voice_name']} · {meta['engine']}**　共 {len(meta['sentences'])} 句，时长 {meta.get('total', 0):.1f} 秒"
          + (f"，生成用时 {meta['sec']} 秒" if meta.get("sec") else "") + f"　随机种子 {meta['params']['seed']}\n\n"
          f"音频：`{meta['wav']}`　字幕：`{os.path.basename(meta['srt'])}`（和音频一起拖进剪映即可）"
          + (f"\n\n第 {'、'.join(map(str, miss))} 句的情感没有对应的参考录音，已用平静录音代替。" if miss else "")
          + (f"\n\n{note}" if note else "")
          + "\n\n👇 点表格里的某一句，可以单独试听、修改或重做这一句。")
    return (meta["wav"], meta["srt"], md, takes.table(meta),
            gr.update(choices=takes.history(), value=meta["id"]))


def on_synth(engine, vid, text, *pv):
    if not vid:
        raise gr.Error("请先选择语音包")
    if not (text or "").strip():
        raise gr.Error("请输入文字")
    params = _params(*pv)
    holder = {}

    def job(log):
        with _queued(log, f"生成语音（{engine}）"):
            holder["meta"] = takes.create(engine, vid, text, params, log)
        return "ok"

    logs = msg = ""
    for logs, msg, _ in stream_task(job):
        if msg is None:
            yield gr.update(), gr.update(), "⏳ 生成中…\n\n" + logs.split("\n")[-1], gr.update(), gr.update(), gr.update()
    if "meta" not in holder:
        yield gr.update(), gr.update(), msg, gr.update(), gr.update(), gr.update()
        return
    m = holder["meta"]
    yield _take_outputs(m) + (m["id"],)


def on_load_take(tid):
    if not tid:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None
    m = takes.load(tid)
    return _take_outputs(m) + (tid,)


def on_pick_sentence(tid, evt: gr.SelectData):
    if not tid:
        return gr.update(), gr.update(), gr.update(), None, ""
    m = takes.load(tid)
    i = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    s = m["sentences"][i]
    return (takes.sentence_path(m, i), s["text"], s["emotion"], i,
            f"已选第 {i + 1} 句（{s.get('dur', 0):.1f} 秒）。不满意可以直接「换个语气重做」，或修改文字 / 情感后「按修改重做」。")


def on_regen(tid, idx, text, emo, use_edit):
    if tid is None or idx is None:
        raise gr.Error("请先在表格里点选一句")
    try:
        with gpu.use("重做一句"):
            m = takes.regen(tid, int(idx), text if use_edit else None, emo if use_edit else None)
    except gpu.GpuBusy as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"重做失败：{e}")
    s = m["sentences"][int(idx)]
    out = _take_outputs(m, note=f"🔁 第 {int(idx) + 1} 句已重做（种子 {s['seed']}），整段音频和字幕已同步更新。")
    return out + (takes.sentence_path(m, int(idx)), s["text"], s["emotion"])


def on_pin(tid, idx):
    """把选中句子用到的参考录音锁定为该语言+情感的默认参考。"""
    if tid is None or idx is None:
        raise gr.Error("请先在表格里点选一句")
    m = takes.load(tid)
    s = m["sentences"][int(idx)]
    if not s.get("ref"):
        raise gr.Error("这句没有记录参考录音（GPT-SoVITS 或旧版本生成的），请用其他引擎重做后再锁定")
    lang = takes._lang(m)
    core.pin_ref(m["vid"], lang, s["emotion"], s["ref"])
    return (f"📌 已锁定：「{m['voice_name']}」以后生成 **{core.LANG_NAMES.get(lang, lang)}·{s['emotion']}** 的句子，"
            f"都用这句的参考录音（`{s['ref']}`）。随机种子仍会带来细微变化。"), voice_info(m["vid"])


def on_cands(tid, idx, text, emo):
    if tid is None or idx is None:
        raise gr.Error("请先在表格里点选一句")
    try:
        with gpu.use("生成候选"):
            paths = takes.candidates(tid, int(idx), 3, text, emo)
    except gpu.GpuBusy as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"生成候选失败：{e}")
    return [gr.update(visible=True)] + paths + [f"🎲 第 {int(idx) + 1} 句的 3 个候选已生成，逐个听，点最像的那个下面的「用这个」。"]


def make_choose(k):
    def f(tid, idx):
        if tid is None or idx is None:
            raise gr.Error("请先在表格里点选一句")
        try:
            m = takes.choose(tid, int(idx), k)
        except ValueError as e:
            raise gr.Error(str(e))
        s = m["sentences"][int(idx)]
        out = _take_outputs(m, note=f"✅ 第 {int(idx) + 1} 句已换成候选 {k + 1}（种子 {s['seed']}），整段音频和字幕已更新。"
                                    "觉得它最像的话，可以再点「📌 锁定」。")
        return out + (takes.sentence_path(m, int(idx)), s["text"], s["emotion"], gr.update(visible=False))
    return f


def on_unpin(vid):
    if not vid:
        raise gr.Error("请先选择语音包")
    core.unpin_all(vid)
    return voice_info(vid)


def on_tlang_change(tlang, engine):
    """粤语只推荐 VoxCPM2：CosyVoice3 / Qwen3-TTS 的粤语被评为「塑料粤语」口音。"""
    if "yue" in TEXT_LANGS.get(tlang, "") and engine in ("CosyVoice3", "Qwen3-TTS") and "VoxCPM2" in ENG.available():
        gr.Info(f"粤语已切换到 VoxCPM2（{engine} 的粤语口音不地道）")
        return gr.update(value="VoxCPM2")
    return gr.update()


def on_save_dict(rows):
    import textnorm
    data = rows.values.tolist() if hasattr(rows, "values") else rows
    n = textnorm.save_rows(data)
    return f"✅ 发音词典已保存（{n} 条），之后生成和重做都会生效。", textnorm.dict_rows() or [["", "", "粤语"]]


def on_delete_take(tid):
    if not tid:
        raise gr.Error("没有选中的作品")
    takes.delete(tid)
    return (None, None, "🗑️ 已删除这条生成记录。", [], gr.update(choices=takes.history(), value=None), None)


def on_compare(vid, text, *pv):
    """同一段文字依次用所有引擎生成（逐个加载，显存只占一个）。"""
    if not vid or not (text or "").strip():
        raise gr.Error("请先选择语音包并输入文字")
    params = _params(*pv)
    outs = {n: (None, "") for n in ENG.ENGINE_NAMES}
    for n in ENG.available():
        try:
            with gpu.use(f"对比试听（{n}）"):
                m = takes.create(n, vid, text, params)
            outs[n] = (m["wav"], f"**{n}**　{m['total']:.1f} 秒，用时 {m['sec']} 秒\n\n{ENG.ENGINE_DESC[n]}")
        except Exception as e:
            outs[n] = (None, f"**{n}** 生成失败：{e}")
        yield [x for n2 in ENG.ENGINE_NAMES for x in outs[n2]]
    ENG.release_all()


def open_outputs():
    os.startfile(core.OUTPUTS)


# ------------------------------------------------------------------ 🎧 试听
PLAY_PAGE = 12
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
SCORES = ["未评", "⭐1", "⭐⭐2", "⭐⭐⭐3", "⭐⭐⭐⭐4", "⭐⭐⭐⭐⭐5"]
TEST_DIR = os.path.join(core.ROOT, "test-voice")


def play_folders():
    """可试听的文件夹：[(显示名, 路径)]"""
    out = [("生成结果（outputs）", core.OUTPUTS)]
    if os.path.isdir(TEST_DIR):
        for root, dirs, files in os.walk(TEST_DIR):
            if any(f.lower().endswith(AUDIO_EXT) for f in files):
                out.append(("test-voice" + root[len(TEST_DIR):], root))
    return out


def _scores_path(folder):
    return os.path.join(folder, "试听评分.json")


def _load_scores(folder):
    try:
        with open(_scores_path(folder), encoding="utf-8") as f:
            return __import__("json").load(f)
    except Exception:
        return {}


def _audio_dur(path):
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None


def play_list(folder, kw, order):
    if not folder or not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXT)]
    for k in (kw or "").split():
        files = [f for f in files if k.lower() in f.lower()]
    if order == "最新在前":
        files.sort(key=lambda f: -os.path.getmtime(os.path.join(folder, f)))
    else:
        files.sort()
    return files


def play_render(folder, kw, order, page):
    files = play_list(folder, kw, order)
    pages = max(1, (len(files) + PLAY_PAGE - 1) // PLAY_PAGE)
    page = int(min(max(1, page or 1), pages))
    cur = files[(page - 1) * PLAY_PAGE: page * PLAY_PAGE]
    sc = _load_scores(folder) if folder else {}
    auto = _load_auto(folder) if folder else {}
    outs = []
    for k in range(PLAY_PAGE):
        if k < len(cur):
            f = cur[k]
            p = os.path.join(folder, f)
            d = _audio_dur(p)
            s = sc.get(f, {})
            tag = "🎙️ 原始录音　" if not f.lower().endswith(".wav") or "人声" in f else ""
            title = f"**{f}**　{tag}{f'{d:.1f} 秒' if d else ''}　{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}"
            outs += [gr.update(visible=True), title, p, s.get("score", "未评"), s.get("note", ""), _auto_md(auto.get(f))]
        else:
            outs += [gr.update(visible=False), "", None, "未评", "", ""]
    info = f"共 {len(files)} 个音频，第 {page}/{pages} 页" if folder else "请选择文件夹"
    return [page, cur, info] + outs + [play_ranking(folder)]


def play_ranking(folder):
    sc = _load_scores(folder) if folder else {}
    rated = sorted(((v.get("score", "未评"), f, v.get("note", "")) for f, v in sc.items() if v.get("score", "未评") != "未评"),
                   key=lambda x: -SCORES.index(x[0]) if x[0] in SCORES else 0)
    if not rated:
        return "还没有评分。听完给每个打星，点「💾 保存本页评分」，这里会按星级排名。"
    rows = ["| 评分 | 文件 | 备注 |", "|---|---|---|"] + [f"| {s} | {f} | {n} |" for s, f, n in rated]
    return "**评分排名**\n\n" + "\n".join(rows)


AUTO_FILE = "自动评分.json"


def _load_auto(folder):
    try:
        with open(os.path.join(folder, AUTO_FILE), encoding="utf-8") as f:
            return __import__("json").load(f)
    except Exception:
        return {}


def _auto_md(r):
    if not r:
        return ""
    sub = "　".join(f"{k} {v}" for k, v in r["sub"].items() if not k.startswith("_"))
    warn = [f for f in r["flags"] if not f.startswith("ℹ️")]
    info = [f for f in r["flags"] if f.startswith("ℹ️")]
    return (f"🤖 **参考 {r['stars']}⭐**（{r['score']} 分）　{sub}"
            + ("\n\n⚠️ " + "；".join(warn) if warn else "\n\n✅ 没发现明显问题")
            + ("\n\n" + "；".join(info) if info else "")
            + ("" if r.get("has_text") else "\n\nℹ️ 找不到原文，没评准确度"))


def play_auto(folder, cur):
    """自动评分本页（只评 wav；结果缓存到该文件夹的 自动评分.json）。"""
    import json
    import autoscore
    if not folder or not cur:
        raise gr.Error("这一页没有音频")
    wavs = [os.path.join(folder, f) for f in cur if f.lower().endswith(".wav") and "人声" not in f]
    if not wavs:
        raise gr.Error("这一页没有生成的 wav 音频")
    try:
        with _queued_sync("自动评分"):
            res = autoscore.evaluate(wavs, log=print)
    except gpu.GpuBusy as e:
        raise gr.Error(str(e))
    auto = _load_auto(folder)
    for r in res:
        auto[os.path.basename(r["wav"])] = r
    with open(os.path.join(folder, AUTO_FILE), "w", encoding="utf-8") as f:
        json.dump(auto, f, ensure_ascii=False, indent=1)
    return [_auto_md(auto.get(f)) if k < len(cur) and (f := cur[k]) else "" for k in range(PLAY_PAGE)] + \
        ["🤖 自动评分只是参考：它能可靠地发现漏读、读错（含数字读法）、明显的长停顿和音色差距过大；"
         "判断不了口音和「语气像不像」。实测与人工评分的秩相关约 0.5。"]


def play_calibrate():
    """汇总所有文件夹里「人工星级 + 自动评分」，校准综合分权重。"""
    import json
    import autoscore
    samples = []
    for _, folder in play_folders():
        sc, auto = _load_scores(folder), _load_auto(folder)
        for f, v in sc.items():
            if f in auto and v.get("score", "未评") != "未评":
                samples.append((auto[f]["sub"], int(v["score"][-1])))
    w, rep = autoscore.calibrate(samples)
    if w:
        with open(os.path.join(core.WORK, "_score_weights.json"), "w", encoding="utf-8") as f:
            json.dump({"weights": w, "report": rep, "n": len(samples), "time": time.strftime("%Y-%m-%d %H:%M")}, f,
                      ensure_ascii=False, indent=1)
    return "📐 " + rep + (f"\n\n新权重：" + "　".join(f"{k} {v:.2f}" for k, v in w.items()) if w else "")


def _queued_sync(task):
    return gpu.use(task)


def play_save(folder, cur, *vals):
    if not folder or not cur:
        raise gr.Error("这一页没有音频")
    import json
    sc = _load_scores(folder)
    scores, notes = vals[:PLAY_PAGE], vals[PLAY_PAGE:]
    for k, f in enumerate(cur):
        if scores[k] != "未评" or (notes[k] or "").strip():
            sc[f] = {"score": scores[k], "note": (notes[k] or "").strip(), "time": time.strftime("%Y-%m-%d %H:%M")}
        else:
            sc.pop(f, None)
    with open(_scores_path(folder), "w", encoding="utf-8") as fo:
        json.dump(sc, fo, ensure_ascii=False, indent=1)
    gr.Info("评分已保存")
    return play_ranking(folder)


# ------------------------------------------------------------------ ⑤ 管理
OV_HEADERS = ["声音", "段数", "已校对", "分钟", "语音包", "VoxCPM2微调", "SoVITS训练", "录音向导段数", "占用MB", "创建时间"]


def disk_md():
    rows = ["| 项目 | 大小 | 说明 |", "|---|---|---|"]
    for k, name, desc, gb in core.disk_usage():
        rows.append(f"| {name} | {gb:.2f} GB | {desc} |")
    return "\n".join(rows)


def manage_refresh():
    choices = project_choices()
    return core.overview(), gr.update(choices=choices), gr.update(choices=choices)


def on_rename(pid, new):
    if not pid:
        raise gr.Error("请先选择要管理的声音")
    try:
        core.rename_project(pid, new)
    except ValueError as e:
        raise gr.Error(str(e))
    return f"✅ 已改名为「{new.strip()}」"


def on_merge(src, dst, ok):
    if not src or not dst or src == dst:
        raise gr.Error("请选择两个不同的声音")
    if not ok:
        raise gr.Error("请先勾选确认")
    a, b = core.load_project(src)["name"], core.load_project(dst)["name"]
    n = core.merge_projects(src, dst)
    return f"✅ 已把「{a}」的 {n} 段素材并入「{b}」，并删除了「{a}」。请对「{b}」重新生成语音包。"


def on_delete(pid, ok):
    if not pid:
        raise gr.Error("请先选择要删除的声音")
    if not ok:
        raise gr.Error("请先勾选「我确认删除」")
    name = core.load_project(pid)["name"]
    core.delete_project(pid)
    return f"🗑️ 已删除「{name}」及其语音包。"


def on_export(pid):
    if not pid or not os.path.exists(os.path.join(core.VOICES, pid, "voice.json")):
        raise gr.Error("这个声音还没有语音包")
    z = core.export_voice(pid)
    return z, f"✅ 已导出：`{z}`（{os.path.getsize(z) / 2 ** 20:.0f} MB）。在另一台装了声音工坊的电脑上用「导入语音包」即可使用。"


def on_import(f):
    if not f:
        raise gr.Error("请先选择语音包 zip 文件")
    try:
        name = core.import_voice(f if isinstance(f, str) else f.name)
    except Exception as e:
        raise gr.Error(f"导入失败：{e}")
    return f"✅ 已导入语音包「{name}」，可以在「④ 文字转语音」里选择使用。"


def on_cleanup(key, ok):
    if not key:
        raise gr.Error("请选择要清理的项目")
    if not ok:
        raise gr.Error("请先勾选确认")
    if gpu.holder():
        raise gr.Error("有任务正在运行，请稍后再清理")
    core.cleanup(key)
    return f"🧹 已清理。", disk_md()


# ------------------------------------------------------------------ 界面
CSS = """
.statusbar {display:flex; flex-wrap:wrap; gap:6px; align-items:center}
/* 状态栏定时刷新时不要变灰闪烁 */
#statusbar, #statusbar *, #statusbar.pending {opacity:1 !important; min-height:0 !important}
#statusbar .wrap, #statusbar .progress-text, #statusbar .eta-bar {display:none !important}
.chip {display:inline-block; padding:2px 10px; border:1px solid var(--border-color-primary); border-radius:999px;
       font-size:13px; background:var(--background-fill-secondary)}
.chip.busy {border-color:#2563eb; color:#1d4ed8}
.chip.warn {border-color:#dc2626; color:#b91c1c; font-weight:600}
.steps {padding:8px 12px; border-radius:10px; background:var(--background-fill-secondary); line-height:2}
.step {padding:2px 8px; border-radius:6px}
.step.done {color:#15803d}
.step.now {background:#dbeafe; color:#1e3a8a; font-weight:600}
.next {font-size:14px; margin-top:2px}
.seg-card {border:1px solid var(--border-color-primary); border-radius:10px; padding:6px 10px; margin-bottom:6px}
#log textarea {font-family: Consolas, monospace; font-size:12px}
.pbar {height:14px; background:var(--background-fill-secondary); border-radius:7px; overflow:hidden;
       border:1px solid var(--border-color-primary)}
.pfill {height:100%; transition:width .6s}
.ptext {font-size:13px; margin-top:4px}
.card {border:2px solid var(--border-color-primary); border-radius:14px; padding:18px 22px; background:var(--background-fill-secondary)}
.card-top {display:flex; justify-content:space-between; align-items:center; margin-bottom:10px}
.emo {color:#fff; padding:3px 14px; border-radius:999px; font-weight:700}
.cnt {font-size:13px; opacity:.8}
.line {font-size:28px; line-height:1.5; font-weight:600; margin:6px 0 10px}
.tip {font-size:14px; opacity:.85}
.recprog {font-size:14px; line-height:2}
.auto-md {font-size:13px}
.hidden-b64 {display:none !important}
.recorder {padding:12px; border:1px solid var(--border-color-primary); border-radius:10px}
.rbtn {padding:8px 18px; border-radius:8px; font-size:16px; font-weight:600; margin-right:8px; cursor:pointer;
       border:1px solid var(--border-color-primary)}
.rbtn.start {background:#dc2626; color:#fff; border-color:#dc2626}
.rbtn:disabled {opacity:.4; cursor:not-allowed}
.rstatus {font-size:14px; margin-left:6px}
"""

# 校对页快捷键：Alt+→ 下一页，Alt+← 上一页，Alt+S 保存，Alt+1…8 播放对应片段
HEAD = """
<script>
document.addEventListener('keydown', (e) => {
  if (!e.altKey) return;
  const click = (id) => { const b = document.getElementById(id); if (b) { b.click(); e.preventDefault(); } };
  if (e.key === 'ArrowRight') click('btn-next');
  else if (e.key === 'ArrowLeft') click('btn-prev');
  else if (e.key === 's' || e.key === 'S') click('btn-save');
  else if (/^[1-8]$/.test(e.key)) {
    const root = document.getElementById('seg-audio-' + e.key);
    if (!root) return;
    const btn = root.querySelector('button.play-pause-button, button[aria-label="Play"], button[aria-label="Pause"], button[aria-label="play-pause"]');
    if (btn) { btn.click(); e.preventDefault(); }
    else { const a = root.querySelector('audio'); if (a) { a.paused ? a.play() : a.pause(); e.preventDefault(); } }
  }
});
</script>
"""

GUIDE = """
**第一次使用？按顺序做就好：**
1. **有 vlog / 录音**：到「① 导入素材」上传，点「🔍 识别说话人」，试听后给每个人起名字 → 自动建好各自的声音。
2. **想录得更准**：到「🎤 录音向导」照着台词念（每种情绪都录几句），文稿自带，不用校对。
3. 到「② 逐段校对」把自动识别的文字改对（先看 🔴 疑点）。
4. 到「③ 语音包与训练」点「生成 / 更新语音包」，素材 5 分钟以上再做 VoxCPM2 微调。
5. 到「④ 文字转语音」输入文案 → 生成音频 + SRT 字幕，拖进剪映即可；不满意的句子可以单独重做。
"""


def build_ui():
    avail = ENG.available()
    with gr.Blocks(title="声音工坊", css=CSS, head=HEAD + RECORDER_JS, theme=gr.themes.Soft()) as demo:
        gname, gtotal, _ = core.gpu_info()
        gr.Markdown(f"# 🎙️ 声音工坊 · 克隆你的声音\n普通话 / 粤语　|　引擎：{' / '.join(avail)}　|　{gname}（{gtotal // 1024} GB 显存）")
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                status = gr.HTML(status_html(), elem_id="statusbar")
            with gr.Column(scale=1, min_width=120):
                release_btn = gr.Button("🧹 释放显存", size="sm")
                cancel_q_btn = gr.Button("取消排队", size="sm")
        with gr.Row(equal_height=False):
            sel_proj = gr.Dropdown(project_choices(), label="当前声音", scale=3)
            new_name = gr.Textbox(label="新建声音", placeholder="写个名字，例如：我的声音", scale=3)
            with gr.Column(scale=1, min_width=120):
                open_btn = gr.Button("打开 / 新建", variant="primary")
        checklist = gr.HTML(checklist_html(None))
        with gr.Accordion("📖 新手指引", open=not core.list_projects()):
            gr.Markdown(GUIDE)
        pid_state = gr.State(None)
        timer = gr.Timer(5)

        # ---------------- ① 导入素材
        with gr.Tab("① 导入素材"):
            with gr.Row():
                with gr.Column(scale=3):
                    files = gr.File(label="上传录音或视频（可多选，不限时长）", file_count="multiple",
                                    file_types=["audio", "video", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".mov"])
                    dur_tip = gr.Markdown("上传后这里会显示总时长和建议。")
                with gr.Column(scale=2):
                    lang = gr.Radio(LANG_CHOICES, value="普通话", label="说话的语言")
                    sep = gr.Checkbox(value=True, label="🎵 去除背景音乐和噪音（人声分离，vlog 推荐）")
                    dereverb = gr.Checkbox(value=False, label="同时去混响（室内回声大时勾选）")
                    analyze_btn = gr.Button("🔍 识别说话人（自动区分单人 / 多人）", variant="primary")
                    direct_btn = gr.Button("直接导入到当前声音（确定只有一个人时）")
                    with gr.Accordion("识别设置", open=False):
                        sens = gr.Slider(0.35, 0.75, value=0.55, step=0.05,
                                         label="合并倾向（把一个人拆成多个时调大；把两个人混成一个时调小）")
                        num_spk = gr.Number(value=0, precision=0, minimum=0, maximum=20, label="已知人数（0 = 自动判断）")
            proc_msg = gr.Markdown()
            with gr.Group(visible=False) as spk_panel:
                gr.Markdown("### 说话人确认\n试听每个人并确认名字：**名字相同会合并成一个声音**；填已有声音的名字会**追加**进去；"
                            "取消「保留」的不处理。🧠 表示已根据声纹自动认出。")
                spk_groups, spk_titles, spk_audios, spk_names, spk_langs, spk_keeps = [], [], [], [], [], []
                for k in range(MAX_SPK):
                    with gr.Group(visible=False, elem_classes="seg-card") as g:
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3):
                                spk_titles.append(gr.Markdown())
                                spk_audios.append(gr.Audio(type="filepath", interactive=False, show_label=False))
                            with gr.Column(scale=2, min_width=200):
                                spk_names.append(gr.Dropdown([], allow_custom_value=True, label="名字（可选已有声音）"))
                                spk_langs.append(gr.Dropdown(LANG_CHOICES, label="语言"))
                                spk_keeps.append(gr.Checkbox(value=True, label="保留，生成语音包"))
                    spk_groups.append(g)
                confirm_btn = gr.Button("✅ 确认并生成语音包", variant="primary")
            diar_state = gr.State(None)
            proc_log = gr.Textbox(label="处理日志", lines=8, max_lines=8, elem_id="log", autoscroll=True)

        # ---------------- 🎤 录音向导
        with gr.Tab("🎤 录音向导") as tab_rec:
            gr.Markdown("照着台词念，每句念完点「保存并下一句」。**文稿自带，不需要校对**；每种情绪都录几句，生成时语气会更丰富。"
                        "建议在安静的房间、离麦克风一拳左右。")
            with gr.Row():
                rec_lang = gr.Radio(["普通话", "粤语"], value="普通话", label="录音语言", scale=2)
                rec_emo = gr.Dropdown(["全部"] + core.EMOTIONS, value="全部", label="只录某种情绪", scale=1)
            rec_prog = gr.HTML()
            rec_card = gr.HTML()
            rec_idx = gr.State(0)
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(RECORDER_HTML)
                    mic = gr.Textbox(elem_id="rec-b64", elem_classes="hidden-b64", show_label=False, container=False)
                    rec_fb = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Row():
                        rec_prev = gr.Button("⏮ 上一句")
                        rec_skip = gr.Button("跳过 ⏭")
                    rec_save_btn = gr.Button("✅ 保存并下一句", variant="primary")
                    with gr.Accordion("录自己的句子", open=False):
                        rec_custom = gr.Textbox(label="自定义台词（填了就用它代替上面的台词）", lines=2)
                        rec_custom_emo = gr.Dropdown(core.EMOTIONS, value="平静", label="这句的情绪")

        # ---------------- ② 逐段校对
        with gr.Tab("② 逐段校对") as tab_check:
            check_stats = gr.Markdown()
            with gr.Row():
                flt = gr.Radio(FILTERS, value="⚠️ 疑点优先", label="筛选（疑点优先：两个识别模型结果差异最大的排前面）", scale=4)
                bulk = gr.Dropdown(core.EMOTIONS, label="本页情感一键设为", scale=1)
            with gr.Row(equal_height=True):
                prev_btn = gr.Button("◀ 上一页", elem_id="btn-prev", size="sm")
                page_num = gr.Number(value=1, show_label=False, container=False, precision=0, minimum=1, min_width=70)
                go_btn = gr.Button("跳转到该页", size="sm")
                next_btn = gr.Button("保存并下一页 ▶", variant="primary", elem_id="btn-next", size="sm")
            page_info = gr.Markdown()
            page_idx = gr.State([])
            groups, titles, audios, texts, langs, emos, keeps, moves = [], [], [], [], [], [], [], []
            for k in range(PAGE):
                with gr.Group(visible=False, elem_classes="seg-card") as grp:
                    titles.append(gr.Markdown())
                    with gr.Row(equal_height=True):
                        audios.append(gr.Audio(type="filepath", interactive=False, show_label=False,
                                               show_download_button=False, scale=3, elem_id=f"seg-audio-{k + 1}"))
                        texts.append(gr.Textbox(show_label=False, placeholder="这段实际说的内容", lines=2, scale=5))
                        with gr.Column(scale=2, min_width=170):
                            langs.append(gr.Dropdown(LANG_CHOICES, show_label=False, container=False))
                            emos.append(gr.Dropdown(core.EMOTIONS, show_label=False, container=False))
                            with gr.Row():
                                keeps.append(gr.Checkbox(label="保留", value=True, min_width=60))
                                moves.append(gr.Dropdown([NO_MOVE], value=NO_MOVE, show_label=False, container=False,
                                                         min_width=100))
                groups.append(grp)
            save_btn = gr.Button("💾 保存本页（标为已校对）", variant="primary", elem_id="btn-save")
            gr.Markdown("⌨️ 快捷键：**Alt + 数字 1～8** 播放第 N 段　**Alt + 右方向键** 保存并下一页　**Alt + 左方向键** 上一页　**Alt + S** 保存本页。"
                        "右下角下拉框可把分错人的片段「移到」其他声音。取消「保留」或文稿为空的片段不参与训练。")

        # ---------------- ③ 语音包与训练
        with gr.Tab("③ 语音包与训练") as tab_train:
            train_stats = gr.Markdown()
            with gr.Row():
                zero_btn = gr.Button("⚡ 生成 / 更新语音包（几秒，三个引擎立即可用）", variant="primary", scale=3)
                stop_btn = gr.Button("⏹ 停止训练", variant="stop", scale=1)
            train_prog = gr.HTML()
            train_msg = gr.Markdown()
            with gr.Group(visible=False) as ab_group:
                gr.Markdown("#### 🎧 训练效果对比（同一句话、同一随机种子）")
                with gr.Row():
                    ab_before = gr.Audio(label="训练前（零样本）", type="filepath", interactive=False)
                    ab_after = gr.Audio(label="训练后", type="filepath", interactive=False)
            with gr.Accordion("🌟 VoxCPM2 LoRA 微调（推荐，效果最好）", open=True):
                gr.Markdown("5 分钟素材约 40 分钟，显存会用到约 7.5GB：**请先关闭剪映、微信、浏览器视频**。"
                            "也可以双击 `训练.bat` 在命令行训练，关掉网页也不会中断。")
                with gr.Row():
                    vox_iters = gr.Slider(200, 2000, value=800, step=50, label="训练步数（按素材时长自动推荐）", scale=3)
                    vox_btn = gr.Button("🎓 开始 VoxCPM2 微调", variant="primary", scale=1)
            with gr.Accordion("GPT-SoVITS 训练（经典，省显存）", open=False):
                with gr.Row():
                    e_sovits = gr.Slider(1, 25, value=8, step=1, label="音色模型训练轮数（推荐 8）")
                    e_gpt = gr.Slider(1, 50, value=15, step=1, label="语气模型训练轮数（推荐 15，过高易吞字）")
                    bs = gr.Slider(1, 12, value=default_batch(), step=1, label="批量大小（显存不足就调小）")
                gsv_btn = gr.Button("🎓 开始 GPT-SoVITS 训练")
            train_log = gr.Textbox(label="训练日志", lines=12, max_lines=12, elem_id="log", autoscroll=True)

        # ---------------- ④ 文字转语音
        with gr.Tab("④ 文字转语音") as tab_tts:
            with gr.Row(equal_height=False):
                voice = gr.Dropdown(voice_choices(), label="语音包", scale=5)
                with gr.Column(scale=1, min_width=90):
                    refresh = gr.Button("🔄 刷新", size="sm")
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    vinfo = gr.Markdown("请选择语音包。")
                with gr.Column(scale=1, min_width=120):
                    unpin_btn = gr.Button("清除锁定", size="sm")
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(label="文案", lines=8,
                                      placeholder="直接输入文案即可，会逐句生成并自动配好字幕。可以用标签控制情感：\n"
                                                  "[开心]今天终于把视频剪完啦！\n[温柔]辛苦大家陪我一起走到这里。\n[悲伤]可惜时间过得太快了……")
                    with gr.Row():
                        tlang = gr.Dropdown(list(TEXT_LANGS.keys()), value="普通话（可夹英文）", label="文案语言")
                        demo_emo = gr.Dropdown(core.EMOTIONS, value="平静", label="默认情感")
                        auto_emo = gr.Checkbox(value=True, label="按句自动判断情感")
                    engine = gr.Radio([(f"{n}：{ENG.ENGINE_DESC[n]}", n) for n in avail], value=avail[0], label="引擎")
                    with gr.Accordion("更多设置", open=False):
                        expr = gr.Slider(0, 1, value=0.6, step=0.05, label="表现力（越高起伏越大；太高可能不稳定）")
                        speed = gr.Slider(0.6, 1.6, value=1.0, step=0.05, label="语速（CosyVoice3 / GPT-SoVITS 有效）")
                        pause = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="句间停顿（秒）")
                        seed = gr.Number(value=-1, precision=0, label="随机种子（-1 为随机；想复现同样的语气就填上次的种子）")
                        multi = gr.Checkbox(value=True, label="多参考融合（GPT-SoVITS 音色更稳定）")
                        strong = gr.Checkbox(value=False, label="VoxCPM2 用文字指令强化情感（情绪更明显，但音色会变淡）")
                        use_lora = gr.Checkbox(value=True, label="使用微调成果（语音包有微调时）")
                    with gr.Row():
                        gen_btn = gr.Button("🔊 生成语音 + 字幕", variant="primary")
                        cmp_btn = gr.Button("🎧 所有引擎对比试听")
                with gr.Column(scale=2):
                    out_audio = gr.Audio(label="整段音频（生成后自动播放）", type="filepath", interactive=False, autoplay=True)
                    srt_file = gr.File(label="SRT 字幕（拖进剪映）", interactive=False, height=90)
                    out_md = gr.Markdown()
                    open_btn2 = gr.Button("📂 打开输出文件夹", size="sm")
            with gr.Accordion("✂️ 逐句编辑 / 历史记录", open=True):
                with gr.Row(equal_height=False):
                    hist = gr.Dropdown(takes.history(), label="生成历史（选择即可载入）", scale=5)
                    with gr.Column(scale=1, min_width=110):
                        del_take = gr.Button("🗑️ 删除这条", size="sm")
                take_state = gr.State(None)
                sent_idx = gr.State(None)
                sent_table = gr.Dataframe(headers=["#", "情感", "文字", "时长(秒)", "开始"], interactive=False, wrap=True,
                                          datatype=["number", "str", "str", "number", "str"])
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        sent_audio = gr.Audio(label="这一句（选中或重做后自动播放）", type="filepath", interactive=False, autoplay=True)
                        sent_hint = gr.Markdown("👆 点表格里的某一句来编辑")
                    with gr.Column(scale=3):
                        sent_text = gr.Textbox(label="文字", lines=2)
                        sent_emo = gr.Dropdown(core.EMOTIONS, label="情感")
                        with gr.Row():
                            regen_btn = gr.Button("🔁 换个语气重做（文字不变）")
                            regen_edit_btn = gr.Button("✏️ 按修改重做", variant="primary")
                        with gr.Row():
                            cand_btn = gr.Button("🎲 生成 3 个候选（听了再选）")
                            pin_btn = gr.Button("📌 这句最像：锁定它的参考录音")
                with gr.Group(visible=False) as cand_group:
                    gr.Markdown("**候选试听**：同一句话、不同随机性。机器判断不了「像不像」（声纹分数和人耳评分几乎不相关），请用耳朵挑。")
                    with gr.Row():
                        cand_audios, cand_picks = [], []
                        for k in range(3):
                            with gr.Column():
                                cand_audios.append(gr.Audio(label=f"候选 {k + 1}", type="filepath", interactive=False))
                                cand_picks.append(gr.Button(f"✅ 用候选 {k + 1}", size="sm"))
            with gr.Accordion("📖 发音词典（纠正读音：用同音字替换，字幕仍显示原文）", open=False):
                gr.Markdown("某个词总读错时，在这里写一个**读音正确的同音写法**。例如粤语「珊姐」的「姐」要读上扬调，可以试「珊者」；"
                            "改完点保存，再对那句点「🔁 重做」试听。数字规则已内置：年份逐位读（2026 → 二零二六），长数字逐位读。")
                import textnorm
                pron = gr.Dataframe(headers=["原文", "读成（同音写法）", "语言"], value=textnorm.dict_rows() or [["", "", "粤语"]],
                                    datatype=["str", "str", "str"], interactive=True, col_count=(3, "fixed"), row_count=(1, "dynamic"))
                with gr.Row():
                    pron_save = gr.Button("💾 保存发音词典", variant="primary")
                    pron_msg = gr.Markdown("语言填：粤语 / 普通话 / 全部")
            with gr.Accordion("🎧 对比试听结果", open=False) as cmp_acc:
                cmp_outs = []
                with gr.Row():
                    for n in ENG.ENGINE_NAMES:
                        with gr.Column():
                            cmp_outs.append(gr.Audio(label=n, type="filepath", interactive=False))
                            cmp_outs.append(gr.Markdown())

        # ---------------- 🎧 试听
        with gr.Tab("🎧 试听") as tab_play:
            gr.Markdown("直接在网页里播放生成的语音和测试音频，可以打分记录哪个最像。评分保存在该文件夹的 `试听评分.json`。")
            with gr.Row(equal_height=False):
                pl_folder = gr.Dropdown(play_folders(), label="文件夹", scale=3)
                pl_kw = gr.Textbox(label="筛选（关键词，空格分隔，如：粤语 _C_）", scale=2)
                pl_order = gr.Radio(["按名称", "最新在前"], value="按名称", label="排序", scale=2)
                with gr.Column(scale=1, min_width=100):
                    pl_refresh = gr.Button("🔄 刷新", size="sm")
            with gr.Row(equal_height=True):
                pl_prev = gr.Button("◀ 上一页", size="sm")
                pl_page = gr.Number(value=1, show_label=False, container=False, precision=0, minimum=1, min_width=70)
                pl_next = gr.Button("下一页 ▶", size="sm")
            pl_info = gr.Markdown()
            pl_cur = gr.State([])
            pl_groups, pl_titles, pl_audios, pl_scores, pl_notes, pl_autos = [], [], [], [], [], []
            for r in range(PLAY_PAGE // 2):
                with gr.Row():
                    for c in range(2):
                        with gr.Group(visible=False, elem_classes="seg-card") as g:
                            pl_titles.append(gr.Markdown())
                            pl_audios.append(gr.Audio(type="filepath", interactive=False, show_label=False))
                            with gr.Row():
                                pl_scores.append(gr.Dropdown(SCORES, value="未评", show_label=False, container=False, min_width=110))
                                pl_notes.append(gr.Textbox(show_label=False, container=False, placeholder="备注：像不像、口音、哪里不对…"))
                            pl_autos.append(gr.Markdown(elem_classes="auto-md"))
                        pl_groups.append(g)
            with gr.Row():
                pl_save = gr.Button("💾 保存本页评分", variant="primary")
                pl_auto_btn = gr.Button("🤖 自动评分本页（参考）")
                pl_cal_btn = gr.Button("📐 用我的评分校准自动评分")
            pl_auto_msg = gr.Markdown()
            pl_rank = gr.Markdown()

        # ---------------- ⑤ 管理
        with gr.Tab("⑤ 管理与清理") as tab_manage:
            ov = gr.Dataframe(headers=OV_HEADERS, value=core.overview(), interactive=False, wrap=True)
            ov_refresh = gr.Button("🔄 刷新", size="sm")
            m_msg = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    m_proj = gr.Dropdown(project_choices(), label="要管理的声音")
                    with gr.Row():
                        m_new = gr.Textbox(label="新名字", scale=3)
                        m_rename = gr.Button("✏️ 改名", scale=1, min_width=80)
                    with gr.Row():
                        m_target = gr.Dropdown(project_choices(), label="合并到", scale=3)
                        m_merge_ok = gr.Checkbox(label="确认合并", scale=1, min_width=80)
                    m_merge = gr.Button("🔗 合并（把上面的声音并入目标）")
                    with gr.Row():
                        m_del_ok = gr.Checkbox(label="我确认删除（素材和语音包都会删除，不可恢复）")
                        m_del = gr.Button("🗑️ 删除", variant="stop")
                with gr.Column():
                    m_export = gr.Button("📦 导出语音包（zip，可备份或换电脑用）")
                    m_zip = gr.File(label="导出的语音包", interactive=False, height=90)
                    m_import_file = gr.File(label="导入语音包（zip）", file_types=[".zip"])
                    m_import = gr.Button("📥 导入")
            gr.Markdown("#### 🧹 磁盘清理")
            d_md = gr.Markdown("点「📏 计算占用」查看各项占用的空间（统计 C 盘缓存需要几秒）。")
            d_calc = gr.Button("📏 计算占用", size="sm")
            with gr.Row():
                d_key = gr.Dropdown([(n, k) for k, n, _, _ in core.disk_items()], label="清理项目", scale=3)
                d_ok = gr.Checkbox(label="确认清理", scale=1, min_width=80)
                d_btn = gr.Button("🧹 清理", scale=1)

        # ================= 事件绑定
        page_outs = [page_num, page_idx, page_info]
        for k in range(PAGE):
            page_outs += [groups[k], titles[k], audios[k], texts[k], langs[k], emos[k], keeps[k], moves[k]]
        slot_vals = texts + langs + emos + keeps + moves
        voice_upd = lambda pid: gr.update(choices=voice_choices(), value=pid if pid and os.path.exists(
            os.path.join(core.VOICES, pid, "voice.json")) else None)

        # 顶部
        timer.tick(status_html, None, [status], show_progress="hidden")
        release_btn.click(on_release, None, [status])
        cancel_q_btn.click(lambda: (gpu.cancel(), gr.Info("已取消本网页里排队中的任务"), status_html())[-1], None, [status])
        open_btn.click(on_open_project, [new_name, sel_proj], [sel_proj, pid_state, new_name])
        sel_proj.select(lambda s: s, [sel_proj], [pid_state])
        pid_state.change(checklist_html, [pid_state], [checklist]) \
            .then(train_info, [pid_state], [train_stats]) \
            .then(suggest_iters, [pid_state], [vox_iters]) \
            .then(voice_upd, [pid_state], [voice]) \
            .then(rec_view, [pid_state, rec_lang, rec_emo, rec_idx], [rec_card, rec_prog, rec_idx]) \
            .then(render_page, [pid_state, page_num, flt], page_outs) \
            .then(stats_md, [pid_state], [check_stats])

        # ① 导入
        files.change(on_files_change, [files], [dur_tip])
        spk_outs = []
        for k in range(MAX_SPK):
            spk_outs += [spk_groups[k], spk_titles[k], spk_audios[k], spk_names[k], spk_langs[k], spk_keeps[k]]
        analyze_btn.click(on_analyze, [files, lang, sep, dereverb, sens, num_spk],
                          [proc_log, proc_msg, diar_state, spk_panel] + spk_outs)
        confirm_btn.click(on_confirm_speakers, [diar_state] + spk_names + spk_langs + spk_keeps,
                          [proc_log, proc_msg, sel_proj, pid_state, spk_panel]) \
            .then(checklist_html, [pid_state], [checklist])
        direct_btn.click(on_direct, [pid_state, files, lang, sep, dereverb], [proc_log, proc_msg]) \
            .then(checklist_html, [pid_state], [checklist])

        # 🎤 录音
        rec_in = [pid_state, rec_lang, rec_emo, rec_idx]
        tab_rec.select(rec_view, rec_in, [rec_card, rec_prog, rec_idx])
        rec_lang.change(lambda p, l, e: rec_view(p, l, e, 0), [pid_state, rec_lang, rec_emo], [rec_card, rec_prog, rec_idx])
        rec_emo.change(lambda p, l, e: rec_view(p, l, e, 0), [pid_state, rec_lang, rec_emo], [rec_card, rec_prog, rec_idx])
        reset_js = "() => { vsRecReset(); }"
        rec_prev.click(lambda p, l, e, i: rec_move(p, l, e, i, -1), rec_in, [rec_card, rec_prog, rec_idx, mic, rec_fb]) \
            .then(None, None, None, js=reset_js)
        rec_skip.click(lambda p, l, e, i: rec_move(p, l, e, i, +1), rec_in, [rec_card, rec_prog, rec_idx, mic, rec_fb]) \
            .then(None, None, None, js=reset_js)
        rec_save_btn.click(rec_save, rec_in + [mic, rec_custom, rec_custom_emo],
                           [rec_card, rec_prog, rec_idx, mic, rec_fb, rec_custom]) \
            .then(None, None, None, js="() => { if (!document.querySelector('#rec-b64 textarea').value) vsRecReset(); }") \
            .then(checklist_html, [pid_state], [checklist])

        # ② 校对
        tab_check.select(render_page, [pid_state, page_num, flt], page_outs).then(stats_md, [pid_state], [check_stats])
        nav_in = [pid_state, page_num, flt, page_idx] + slot_vals
        for btn, d in ((prev_btn, -1), (next_btn, +1), (go_btn, 0), (save_btn, 0)):
            btn.click(make_nav(d), nav_in, page_outs + [check_stats]).then(checklist_html, [pid_state], [checklist])
        flt.change(lambda p, f: render_page(p, 1, f), [pid_state, flt], page_outs)
        bulk.select(bulk_emotion, [page_idx, bulk], emos)

        # ③ 训练
        tab_train.select(train_info, [pid_state], [train_stats])
        zero_btn.click(on_zero, [pid_state], [train_log, train_msg]) \
            .then(train_info, [pid_state], [train_stats]).then(checklist_html, [pid_state], [checklist]) \
            .then(voice_upd, [pid_state], [voice])
        ab_outs = [train_log, train_msg, train_prog, ab_group, ab_before, ab_after]
        vox_btn.click(on_train_vox, [pid_state, vox_iters], ab_outs) \
            .then(train_info, [pid_state], [train_stats]).then(checklist_html, [pid_state], [checklist])
        gsv_btn.click(on_train_gsv, [pid_state, e_sovits, e_gpt, bs], ab_outs) \
            .then(train_info, [pid_state], [train_stats]).then(checklist_html, [pid_state], [checklist])
        stop_btn.click(on_stop, None, [train_msg])

        # ④ 合成
        refresh.click(lambda: gr.update(choices=voice_choices()), None, [voice])
        tab_tts.select(lambda: (gr.update(choices=voice_choices()), gr.update(choices=takes.history())), None, [voice, hist])
        voice.change(voice_info, [voice], [vinfo])
        pv = [tlang, demo_emo, auto_emo, expr, speed, pause, seed, multi, strong, use_lora]
        take_outs = [out_audio, srt_file, out_md, sent_table, hist, take_state]
        gen_btn.click(on_synth, [engine, voice, text] + pv, take_outs).then(checklist_html, [pid_state], [checklist])
        hist.select(on_load_take, [hist], take_outs)
        sent_table.select(on_pick_sentence, [take_state], [sent_audio, sent_text, sent_emo, sent_idx, sent_hint])
        regen_outs = [out_audio, srt_file, out_md, sent_table, hist, sent_audio, sent_text, sent_emo]
        regen_btn.click(lambda t, i, x, e: on_regen(t, i, x, e, False), [take_state, sent_idx, sent_text, sent_emo], regen_outs)
        regen_edit_btn.click(lambda t, i, x, e: on_regen(t, i, x, e, True), [take_state, sent_idx, sent_text, sent_emo], regen_outs)
        pin_btn.click(on_pin, [take_state, sent_idx], [sent_hint, vinfo])
        cand_btn.click(on_cands, [take_state, sent_idx, sent_text, sent_emo], [cand_group] + cand_audios + [sent_hint])
        for k in range(3):
            cand_picks[k].click(make_choose(k), [take_state, sent_idx], regen_outs + [cand_group])
        unpin_btn.click(on_unpin, [voice], [vinfo])
        tlang.change(on_tlang_change, [tlang, engine], [engine])
        pron_save.click(on_save_dict, [pron], [pron_msg, pron])
        del_take.click(on_delete_take, [take_state], take_outs)
        open_btn2.click(open_outputs, None, None)
        cmp_btn.click(lambda: gr.update(open=True), None, [cmp_acc]).then(on_compare, [voice, text] + pv, cmp_outs)

        # 🎧 试听
        pl_outs = [pl_page, pl_cur, pl_info]
        for k in range(PLAY_PAGE):
            pl_outs += [pl_groups[k], pl_titles[k], pl_audios[k], pl_scores[k], pl_notes[k], pl_autos[k]]
        pl_outs.append(pl_rank)
        pl_in = [pl_folder, pl_kw, pl_order]
        tab_play.select(lambda f: gr.update(choices=play_folders(), value=f or play_folders()[0][1]), [pl_folder], [pl_folder])
        pl_folder.change(lambda f, k, o: play_render(f, k, o, 1), pl_in, pl_outs)
        pl_kw.submit(lambda f, k, o: play_render(f, k, o, 1), pl_in, pl_outs)
        pl_order.change(lambda f, k, o: play_render(f, k, o, 1), pl_in, pl_outs)
        pl_refresh.click(lambda f, k, o, p: play_render(f, k, o, p), pl_in + [pl_page], pl_outs)
        pl_prev.click(lambda f, k, o, p: play_render(f, k, o, (p or 1) - 1), pl_in + [pl_page], pl_outs)
        pl_next.click(lambda f, k, o, p: play_render(f, k, o, (p or 1) + 1), pl_in + [pl_page], pl_outs)
        pl_save.click(play_save, [pl_folder, pl_cur] + pl_scores + pl_notes, [pl_rank])
        pl_auto_btn.click(play_auto, [pl_folder, pl_cur], pl_autos + [pl_auto_msg])
        pl_cal_btn.click(play_calibrate, None, [pl_auto_msg])

        # ⑤ 管理
        m_refresh_outs = [ov, m_proj, m_target]
        d_calc.click(disk_md, None, [d_md])
        tab_manage.select(manage_refresh, None, m_refresh_outs)
        ov_refresh.click(manage_refresh, None, m_refresh_outs)
        after = lambda ev: ev.then(manage_refresh, None, m_refresh_outs) \
            .then(lambda: gr.update(choices=project_choices()), None, [sel_proj])
        after(m_rename.click(on_rename, [m_proj, m_new], [m_msg]))
        after(m_merge.click(on_merge, [m_proj, m_target, m_merge_ok], [m_msg]))
        after(m_del.click(on_delete, [m_proj, m_del_ok], [m_msg]))
        m_export.click(on_export, [m_proj], [m_zip, m_msg])
        after(m_import.click(on_import, [m_import_file], [m_msg]))
        d_btn.click(on_cleanup, [d_key, d_ok], [m_msg, d_md])

        demo.load(rec_view, rec_in, [rec_card, rec_prog, rec_idx])
    return demo


if __name__ == "__main__":
    import atexit
    atexit.register(ENG.release_all)
    atexit.register(core.stop_all)
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("STUDIO_PORT", 7866)),
        inbrowser=not os.environ.get("STUDIO_NO_BROWSER"),
        allowed_paths=[core.WORK, core.VOICES, core.OUTPUTS, TEST_DIR], show_api=False)
