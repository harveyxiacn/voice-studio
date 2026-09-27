# -*- coding: utf-8 -*-
"""录音向导：按「情感 × 台词」逐句录音。文稿自带（无需识别和校对），并检查录音质量。"""
import os
import time

import numpy as np

import core

TIPS = {
    "平静": "正常说话的语气，像在跟朋友介绍事情。",
    "开心": "嘴角上扬，带着笑意说。",
    "兴奋": "音量和语速都提高一点，像刚看到惊喜。",
    "温柔": "放轻声音、放慢语速，像在哄人或安慰人。",
    "悲伤": "声音低一点、慢一点，带点叹气。",
    "生气": "加重语气，像真的有点火大（不用吼）。",
    "惊讶": "语调上扬，像听到意想不到的消息。",
    "严肃": "咬字清楚、语气坚定，像在提醒重要的事。",
}

SCRIPTS = {
    "普通话": {
        "平静": ["今天我们来到了一个安静的小镇，街道两边都是老房子。",
                 "早上七点起床，简单吃了个早餐，就出发去海边了。",
                 "这家店的招牌菜是红烧肉，据说已经做了三十多年。",
                 "从酒店走到地铁站大概十分钟，路上有很多便利店。",
                 "接下来给大家看看我们今天住的房间，空间还挺大的。",
                 "天气预报说下午会下雨，所以我们带了两把伞。"],
        "开心": ["太好了，终于吃到心心念念的那家火锅啦！",
                 "哈哈，你看这只小狗一直跟着我们，好可爱呀。",
                 "今天的日落真的太美了，感觉一天的累都值了。",
                 "收到大家的留言我真的特别开心，谢谢你们的支持！",
                 "我们抽中了免费升级房间，简直是今天最好的消息。",
                 "这杯奶茶甜度刚刚好，我觉得可以再来一杯。"],
        "兴奋": ["天哪！你们快看，那边有好多烟花！",
                 "我们终于登顶了！风景也太壮观了吧！",
                 "中了中了！我居然抽到了一等奖！",
                 "快快快，演唱会马上就要开始了，冲啊！",
                 "这是我第一次坐热气球，心跳得好快！",
                 "进球了！这个进球实在是太漂亮了！"],
        "温柔": ["没关系的，慢慢来，我们不着急。",
                 "晚安啦，今天辛苦了，好好休息吧。",
                 "别怕，我一直都在你身边陪着你。",
                 "天冷了，记得多穿一件外套，别着凉了。",
                 "谢谢你愿意听我说这些，真的很感动。",
                 "宝贝，生日快乐，愿你每天都开开心心的。"],
        "悲伤": ["唉，最后一天了，真的有点舍不得离开这里。",
                 "可惜那天下雨，没能看到传说中的日出。",
                 "小时候常来的那家店，今年已经关门了。",
                 "看着这些旧照片，突然好想念外婆。",
                 "我们排了两个小时的队，结果还是卖完了。",
                 "旅行总是过得特别快，一眨眼就要回家了。"],
        "生气": ["这也太过分了吧，排了这么久居然插队！",
                 "说好的海景房，结果窗外全是工地！",
                 "我已经说了三遍了，你怎么还是不听！",
                 "司机绕了一大圈路，还多收了我们五十块！",
                 "真是受够了，这个快递又送错地址了。",
                 "凭什么不给退款，明明是你们的问题！"],
        "惊讶": ["什么？这么大一碗面才十块钱？",
                 "不会吧，你居然也在这里旅游？",
                 "真的假的？这里以前竟然是一片海？",
                 "哇，原来这座桥已经有一千年的历史了！",
                 "你说什么？航班取消了？",
                 "咦，我的手机怎么突然没电了？"],
        "严肃": ["请大家注意，过马路的时候一定要看红绿灯。",
                 "出发前务必检查好证件和充电器。",
                 "这里是自然保护区，禁止乱扔垃圾。",
                 "如果遇到紧急情况，请马上拨打报警电话。",
                 "山路比较危险，请大家不要走到护栏外面。",
                 "重要的事情说三遍：带好护照，带好护照，带好护照。"],
    },
    "粤语": {
        "平静": ["今日我哋嚟到一个好静嘅小镇，两边都系旧屋。",
                 "朝早七点起身，食咗个早餐就去海边喇。",
                 "呢间铺头嘅招牌菜系红烧肉，听讲做咗三十几年。",
                 "由酒店行去地铁站大概十分钟，路上有好多便利店。",
                 "跟住畀大家睇下我哋今日住嘅房，都几大下。",
                 "天气预报话下昼会落雨，所以我哋带咗两把遮。"],
        "开心": ["好嘢，终于食到我成日谂住嗰间火锅喇！",
                 "哈哈，你睇呢只狗仔一路跟住我哋，好得意呀。",
                 "今日嘅日落真系靓到爆，成日嘅攰都值晒。",
                 "收到大家嘅留言我真系好开心，多谢你哋支持！",
                 "我哋抽中免费升级房，真系今日最好嘅消息。",
                 "呢杯奶茶啱啱好甜，我觉得可以再饮多杯。"],
        "兴奋": ["哇！你哋快啲睇，嗰边有好多烟花呀！",
                 "我哋终于行到山顶喇！啲风景劲到震！",
                 "中咗呀！我竟然抽到头奖！",
                 "快啲快啲，演唱会就嚟开始喇，冲呀！",
                 "我第一次坐热气球，个心跳到好快！",
                 "入波喇！呢球真系入得好靓！"],
        "温柔": ["唔紧要嘅，慢慢嚟，我哋唔使急。",
                 "早唞啦，今日辛苦晒，好好休息下。",
                 "唔使惊，我一直都喺你身边陪住你。",
                 "天冻喇，记得着多件褛，唔好冷亲。",
                 "多谢你肯听我讲呢啲，我真系好感动。",
                 "宝贝，生日快乐，希望你日日都开开心心。"],
        "悲伤": ["唉，最后一日喇，真系有啲唔舍得走。",
                 "可惜嗰日落雨，睇唔到传说中嘅日出。",
                 "细个成日去嗰间铺，今年已经执咗笠。",
                 "睇住呢啲旧相，突然好挂住婆婆。",
                 "我哋排咗两个钟，结果都系卖晒。",
                 "去旅行啲时间总系过得特别快，一眨眼就要返屋企。"],
        "生气": ["呢个都太过分喇啩，排咗咁耐竟然有人打尖！",
                 "讲好系海景房，点知出面成个地盘！",
                 "我已经讲咗三次喇，你点解仲系唔听！",
                 "个司机兜咗个大圈，仲多收咗我哋五十蚊！",
                 "真系顶唔顺，个速递又送错地址。",
                 "凭咩唔畀退款，明明系你哋嘅问题！"],
        "惊讶": ["咩话？咁大碗面先至十蚊？",
                 "唔系掛，你都嚟咗呢度旅行？",
                 "真定假㗎？呢度以前竟然系一片海？",
                 "哇，原来呢条桥已经有一千年历史！",
                 "你讲咩话？班机取消咗？",
                 "咦，我部手机点解突然冇电嘅？"],
        "严肃": ["请大家留意，过马路一定要睇红绿灯。",
                 "出发之前记住检查好证件同充电器。",
                 "呢度系自然保护区，唔准乱掉垃圾。",
                 "如果遇到紧急情况，请即刻打电话报警。",
                 "山路比较危险，请大家唔好行出栏杆外面。",
                 "重要嘅嘢讲三次：带定护照，带定护照，带定护照。"],
    },
}


def script_list(lang, emotion="全部"):
    """[(脚本ID, 情感, 台词)]"""
    out = []
    for emo, lines in SCRIPTS[lang].items():
        if emotion not in ("全部", emo):
            continue
        for i, t in enumerate(lines):
            out.append((f"{core.LANGS[lang]}-{emo}-{i}", emo, t))
    return out


def quality(y, sr):
    """录音质量检查，返回 (是否可用, 提示文字, 裁掉首尾静音后的音频)。"""
    if y.ndim > 1:
        y = y.mean(1)
    hop = sr // 100
    n = len(y) // hop
    if n < 30:
        return False, "❌ 录音太短，请重新录。", y
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    noise = np.percentile(db, 10)
    speech = np.percentile(db, 95)
    peak = float(np.abs(y).max())
    clip = float((np.abs(y) > 0.99).mean())
    # 裁掉首尾静音（保留 0.15 秒余量）
    act = np.where(db > max(noise + 10, speech - 35))[0]
    if len(act):
        b = max(0, (act[0] - 15) * hop)
        e = min(len(y), (act[-1] + 15) * hop)
        y = y[b:e]
    dur = len(y) / sr
    issues, ok = [], True
    if dur < 1.0 or speech < -55:
        return False, "❌ 没录到声音或太短，请检查麦克风是否选对、是否被静音，然后重录。", y
    if speech < -38:
        issues.append("🔈 声音太小，请靠近麦克风或调大音量")
        ok = speech >= -48
    if clip > 0.001 or peak >= 0.999:
        issues.append("💥 有爆音，请离麦克风远一点或调小音量")
        ok = ok and clip < 0.01
    if speech - noise < 15:
        issues.append(f"🌫️ 背景噪音偏大（信噪比 {speech - noise:.0f}dB），请换个安静的地方")
        ok = ok and speech - noise >= 8
    if dur > 14:
        issues.append("⏱️ 这句超过 14 秒，建议分成两句录")
    level = "✅ 录音质量良好" if not issues else ("⚠️ 可以使用，但" if ok else "❌ 质量较差，建议重录：")
    return ok, f"{level}{'，'.join(issues)}（时长 {dur:.1f} 秒，音量 {speech:.0f}dB，信噪比 {speech - noise:.0f}dB）", y


def save_recording(pid, audio_path, text, emotion, lang_name, script_id=""):
    """保存一句录音为已校对的切片。同一台词重录会覆盖上一次。返回 (是否成功, 提示)。"""
    from scipy.io import wavfile
    y = core._load_audio(audio_path)
    ok, msg, y = quality(y, core.SR)
    if not ok:
        return False, msg
    m = float(np.abs(y).max())
    if m > 0:
        y = y / m * 0.9
    proj = core.load_project(pid)
    rec = proj.setdefault("recorded", {})
    old = rec.get(script_id) if script_id else None
    if old:  # 重录：删除上一次
        proj["segments"] = [s for s in proj["segments"] if s["file"] != old]
        try:
            os.remove(os.path.join(core.project_dir(pid), "slices", old))
        except OSError:
            pass
    name = f"rec_{time.strftime('%m%d%H%M%S')}_{len(proj['segments']):04d}.wav"
    wavfile.write(os.path.join(core.project_dir(pid), "slices", name), core.SR, (y * 32767).astype(np.int16))
    proj["segments"].append({
        "file": name, "dur": round(len(y) / core.SR, 2), "lang": core.LANGS[lang_name],
        "text": text.strip(), "asr": text.strip(), "alt": "", "conf": 1.0, "emotion": emotion,
        "keep": True, "checked": True, "source": "录音向导",
    })
    if script_id:
        rec[script_id] = name
    core.save_project(proj)
    return True, msg


def progress(pid, lang):
    """已录进度：(已录数, 总数, {情感: (已录, 总)})"""
    rec = core.load_project(pid).get("recorded", {}) if pid else {}
    per = {}
    for sid, emo, _ in script_list(lang):
        d, t = per.get(emo, (0, 0))
        per[emo] = (d + (sid in rec), t + 1)
    done = sum(d for d, _ in per.values())
    return done, sum(t for _, t in per.values()), per
