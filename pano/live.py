#!/usr/bin/env python3
"""X5 全景 + 跟随 + 人物信息标注 —— 本地网页。

    .venv/bin/python -m pano.live                  # http://127.0.0.1:8790
    .venv/bin/python -m pano.live --port 8790
    .venv/bin/python -m pano.live --selftest       # 不开服务，离线渲一张图自检

数据源
------
默认从 x5-link 的 `http://127.0.0.1:8765/snapshot.jpg` 取真帧。
**如果 X5 当前没有出帧**（`/status` 里 `last_frame_age_s` 很大 —— 相机拔了/关机会
这样，而 x5-link 仍在把最后一帧当缓存发出来），就自动退回 `pano/demo/real_x5/`
里那张实拍帧继续把整条流水线跑起来，并在页面上**明确标出当前来源**。
相机接回来之后不用重启，只要 `/status` 的帧龄变小，就会自动切回实时。

为什么要有这一页
----------------
`pano/__main__.py` 是离线的（跑一个文件夹出一批图），看不到"正在跟"的感觉。
这一页把整条链路做成常驻循环，并且把结果**标注在平面上**：

    等距柱状全景 ──► 找人/跟住（yaw/pitch） ──► 抽出该方位的透视画面
                                                 └─► GestureMate 动作检测 + 语义
    把 yaw/pitch 再画回全景平面  ──►  在人的位置上标出他的各种信息

⚠️ 这是**第 ① 步（跟随）+ 一个临时的信息标注视图**。
产品要的"在一个平面当中给这个人标信息"选的是 **AR 式贴到真实房间的地面/墙面**，
那一步还没做 —— 它先要估出重力方向（X5 在桌上时画面的"上"≠重力的"上"，
实测人体轴偏了约 20°）。这一页标在**全景展开平面**上，是能看到信息、但还不是
贴到房间平面上的版本。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "webui"))

from pano.follow import PanoFollower, annotate               # noqa: E402
from pano.projection import view_point_to_pano               # noqa: E402

FALLBACK_FRAME = os.path.join(_HERE, "demo/real_x5", "0_X5实机原始帧_2880x1440.jpg")

DEFAULT_X5 = "http://127.0.0.1:8765"
PANO_W, PANO_H = 1440, 720          # 输出全景平面的尺寸
VIEW_OUT = 640                      # 输出跟随视角的尺寸


# ─────────────────────────────────────────────── 画在"平面"上的东西

def _px(yaw: float, pitch: float, w: int, h: int):
    """全景角度 -> 等距柱状平面上的像素。与 projection.py 的约定一致。"""
    x = (yaw / 360.0 + 0.5) * (w - 1)
    y = (0.5 - pitch / 180.0) * (h - 1)
    return int(round(x)), int(round(y))


def _frustum_outline(result, n: int = 10):
    """把当前视角的四条边采成一串全景角度，用来在全景平面上画出"我正在看哪儿"。"""
    view = result.view
    if view is None:
        return []
    vh, vw = view.shape[:2]
    border = []
    for i in range(n + 1):
        t = i / n
        border += [(t, 0.0), (1.0, t), (1.0 - t, 1.0), (0.0, 1.0 - t)]
    us = np.array([u * vw for u, _ in border])
    vs = np.array([v * vh for _, v in border])
    ys, ps = view_point_to_pano(us, vs, vw, vh, result.fix.fov,
                                result.fix.yaw, result.fix.pitch)
    return list(zip(np.asarray(ys).ravel().tolist(), np.asarray(ps).ravel().tolist()))


def draw_pano(erp, result, w=PANO_W, h=PANO_H):
    """等距柱状平面 + 跟随结果标注（这是"把信息标在平面上"的那个平面）。"""
    img = cv2.resize(erp, (w, h), interpolation=cv2.INTER_AREA)
    f = result.fix

    # 参考线：赤道 + 每 90° 一根经线
    cv2.line(img, (0, (h - 1) // 2), (w - 1, (h - 1) // 2), (110, 110, 110), 1)
    for yaw in (-180, -90, 0, 90, 180):
        x, _ = _px(float(yaw), 0.0, w, h)
        cv2.line(img, (x, 0), (x, h - 1), (95, 95, 95), 1)
        cv2.putText(img, f"{yaw:+d}", (max(2, x + 3), 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

    if not f.locked:
        cv2.rectangle(img, (0, 0), (w - 1, 30), (0, 0, 0), -1)
        cv2.putText(img, "NO LOCK - sweeping for a person...", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        return img

    # 正在看的那块（视锥轮廓）。画在单独的图层上半透明合成 —— 锁定的俯仰角较高时，
    # 视锥的角会越过天顶(±90°)，投影到平面上会自己绕回来，线条本来就"花"；
    # 压成半透明只是让它读起来像"我看的范围"，而不是画错。
    pts = _frustum_outline(result, n=6)
    overlay = img.copy()
    poly, prev = [], None
    for yaw, pitch in pts:
        p = _px(yaw, pitch, w, h)
        if prev is not None and abs(p[0] - prev[0]) > w * 0.5:
            if len(poly) > 1:                      # 跨 ±180 断开，别画横穿整幅的线
                cv2.polylines(overlay, [np.array(poly)], False, (255, 210, 90), 2, cv2.LINE_AA)
            poly = []
        poly.append(p)
        prev = p
    if len(poly) > 1:
        cv2.polylines(overlay, [np.array(poly)], False, (255, 210, 90), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # 人的位置
    mx, my = _px(f.yaw, f.pitch, w, h)
    cv2.circle(img, (mx, my), 22, (60, 255, 120), 3, cv2.LINE_AA)
    cv2.drawMarker(img, (mx, my), (60, 255, 120), cv2.MARKER_CROSS, 34, 2, cv2.LINE_AA)

    # 信息卡片（贴在他的位置上）
    lm = result.landmarks or {}
    fy = result.features or {}
    body = (fy.get("body") or {})
    hu = body.get("handsUp") or {}
    lines = [
        f"PERSON   score {f.score:.2f}",
        f"yaw {f.yaw:+.1f}  pitch {f.pitch:+.1f}  fov {f.fov:.0f}",
        f"face {len(lm.get('face') or [])}  body {len(lm.get('body') or [])}",
        f"blinks {fy.get('blinkCount', '-')}  "
        f"handsUp L{int(bool(hu.get('left')))} R{int(bool(hu.get('right')))}",
    ]
    head = fy.get("head") or {}
    if head:
        lines.append(f"head roll {head.get('rollDeg', 0):.0f}deg")
    bw, bh = 232, 16 * len(lines) + 12
    bx = min(max(4, mx + 26), w - bw - 4)
    by = min(max(4, my - bh // 2), h - bh - 4)
    box = img[by:by + bh, bx:bx + bw]
    cv2.addWeighted(box, 0.25, np.zeros_like(box), 0.0, 0, box)   # 压暗一点好读
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 255, 120), 1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (bx + 8, by + 18 + 16 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 255, 240), 1, cv2.LINE_AA)
    cv2.line(img, (mx, my), (bx + (0 if mx < bx else bw), by + bh // 2),
             (60, 255, 120), 1, cv2.LINE_AA)
    return img


def draw_view(result):
    """跟随视角（就是"玩家视角"）：骨架/脸/手 + 语义，直接复用离线那套 annotate。"""
    if not result.fix.locked or result.view is None:
        img = np.full((VIEW_OUT, VIEW_OUT, 3), 24, np.uint8)
        cv2.putText(img, "NO LOCK", (VIEW_OUT // 2 - 78, VIEW_OUT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (70, 70, 200), 2, cv2.LINE_AA)
        return img
    img = annotate(result)
    return cv2.resize(img, (VIEW_OUT, VIEW_OUT), interpolation=cv2.INTER_LINEAR)


def status_payload(result, fps, source, note):
    f = result.fix
    lm = result.landmarks or {}
    fy = result.features or {}
    body = fy.get("body") or {}
    hu = body.get("handsUp") or {}
    head = fy.get("head") or {}
    mouth = fy.get("mouth") or {}
    # "锁着但这一帧根本没检出人"要说出来，别装作在跟 —— 跟丢计数就是它。
    if f.locked and f.misses > 0:
        state = f"跟丢中（连续 {f.misses} 帧没检出人）"
    else:
        state = "已锁定" if f.locked else "未找到人"
    rows = [
        ["状态", state],
        ["方位 yaw", f"{f.yaw:+.1f}°"],
        ["俯仰 pitch", f"{f.pitch:+.1f}°"],
        ["视场 fov", f"{f.fov:.0f}°"],
        ["置信 score", f"{f.score:.2f}"],
        ["脸 / 身体", f"{len(lm.get('face') or [])} / {len(lm.get('body') or [])}"],
    ]
    if fy.get("available"):
        rows += [
            ["眨眼次数", str(fy.get("blinkCount", "-"))],
            ["头朝向", f"{head.get('direction', '-')}（roll {head.get('rollDeg', 0):.0f}°）"],
            ["嘴巴", f"{mouth.get('state', '-')}（开合 {mouth.get('open', 0):.2f}）"],
            ["举手", f"左 {'是' if hu.get('left') else '否'} / 右 {'是' if hu.get('right') else '否'}"],
            ["身体", f"{body.get('lean', '-')}（肩线 {body.get('shoulderTiltDeg', 0):.0f}°）"],
        ]
    rows += [
        ["定位 / 检测", f"{result.t_locate:.0f} ms / {result.t_detect:.0f} ms"],
        ["循环帧率", f"{fps:.1f} FPS"],
        ["数据源", source],
    ]
    return {"locked": bool(f.locked), "rows": rows, "note": note}


# ─────────────────────────────────────────────── 取帧

class StreamReader(threading.Thread):
    """常驻读 x5-link 的 MJPEG，永远只保留"最新的一帧"。

    为什么要它，而不是每帧 HTTP 抓一次 `snapshot.jpg`：抓一次要重新建连接、
    重新下 324KB，实测 **20–30ms**，而且这段时间跟随线程只能干等；再算上解码
    （8–15ms），一帧 100ms 的预算里光"取到一张图"就占了 30–45ms。
    把"取帧 + 解码"整个挪到后台线程之后，主循环拿到手的直接就是 ndarray，
    这一整段开销**从关键路径上消失了**（只在主循环之外并行发生）。

    限速解码（`max_fps`）是为了别为了 15 FPS 的消费端去解 30 FPS 的帧 —— 白烧 CPU。
    """

    def __init__(self, url: str, max_fps: float = 25.0):
        super().__init__(daemon=True)
        self.url = url
        self.max_fps = max(1.0, float(max_fps))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self.frames = 0
        self.error: str | None = None

    def run(self):
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(self.url,
                                             headers={"Cache-Control": "no-store"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        buf += chunk
                        if len(buf) > 8 << 20:          # 异常数据别把内存吃光
                            buf = buf[-1 << 20:]
                        while True:
                            i = buf.find(b"\xff\xd8")
                            j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
                            if i < 0 or j < 0:
                                break
                            jpg = buf[i:j + 2]
                            buf = buf[j + 2:]
                            now = time.time()
                            with self._lock:
                                if now - self._t < 1.0 / self.max_fps:
                                    continue
                            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                                 cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self._lock:
                                    self._frame = frame
                                    self._t = now
                                    self.frames += 1
                                self.error = None
            except Exception as e:
                self.error = repr(e)
                time.sleep(1.0)

    def latest(self):
        """返回 (时间戳, 帧) 或 None。永不阻塞。"""
        with self._lock:
            if self._frame is None:
                return None
            return self._t, self._frame

    def stop(self):
        self._stop.set()


class Source:
    """取一帧等距柱状图。X5 没出帧就退回离线实拍帧。"""

    def __init__(self, x5_url: str):
        self.x5 = x5_url.rstrip("/")
        self.offline = cv2.imread(FALLBACK_FRAME) if os.path.exists(FALLBACK_FRAME) else None
        self.mode = "x5"
        self._last_status = 0.0
        self._age = None

    async def _x5_age(self, session):
        """读 x5-link 的 /status，拿"最后一帧多久以前" —— 这才是它活没活的判据。"""
        now = time.time()
        if now - self._last_status < 2.0:
            return self._age
        self._last_status = now
        try:
            async with session.get(f"{self.x5}/status", timeout=3) as r:
                j = json.loads(await r.text())
            self._age = float(j.get("last_frame_age_s", 1e9))
        except Exception:
            self._age = None
        return self._age

    async def get(self, session):
        age = await self._x5_age(session)
        live_ok = age is not None and age < 8.0
        if live_ok:
            try:
                async with session.get(f"{self.x5}/snapshot.jpg", timeout=5) as r:
                    buf = await r.read()
                frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.mode = "x5"
                    return frame, "X5 实时", ""
            except Exception:
                pass
        self.mode = "offline"
        if self.offline is None:
            return None, "无", "X5 未出帧，且找不到离线实拍帧"
        age_txt = "未知" if age is None else f"{age/60:.0f} 分钟"
        return (self.offline.copy(), "离线实拍帧",
                f"X5 相机当前没有出帧（最后一帧在 {age_txt} 前），"
                f"当前用 pano/demo/real_x5 里的实拍帧演示；把相机接回来刷新即可变回实时。")


# ─────────────────────────────────────────────── 常驻循环 + 网页

class Hub:
    def __init__(self):
        self._imgs: dict[str, tuple[int, bytes]] = {}
        self.status: dict = {"locked": False, "rows": [], "note": "启动中…"}

    def put(self, key: str, jpg: bytes):
        seq = self._imgs.get(key, (0, b""))[0]
        self._imgs[key] = (seq + 1, jpg)

    def get(self, key: str):
        return self._imgs.get(key, (0, None))

    def snapshot(self, key: str):
        return self._imgs.get(key, (0, None))[1]


async def worker(hub: Hub, args):
    import aiohttp
    src = Source(args.x5)
    fol = PanoFollower(detect_every=args.detect_every)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    # fps<=0 表示不人为限速（由取流线程的 max_fps 兜底）
    period = 0.0 if args.fps <= 0 else 1.0 / args.fps
    n = 0
    t_prev = time.time()

    # 取帧+解码挪到后台：主循环的关键路径只剩 跟随 + 检测 + 画图
    reader = StreamReader(f"{args.x5}/stream.mjpg",
                          max_fps=max(20.0, args.fps * 1.5))
    reader.start()
    stale_after = 3.0

    async with aiohttp.ClientSession() as session:
        while True:
            t0 = time.time()
            got = reader.latest()
            if got is not None and (t0 - got[0]) < stale_after:
                frame, source_name, note = got[1], "X5 实时（常驻取流）", ""
            else:
                # 取流没起来 / 相机没出帧 -> 退回原来的 snapshot 轮询与离线帧
                frame, source_name, note = await src.get(session)
            if frame is None:
                hub.status = {"locked": False, "rows": [], "note": note}
                await asyncio.sleep(1.0)
                continue
            try:
                result = await asyncio.to_thread(fol.step, frame, True)
                pano = await asyncio.to_thread(draw_pano, frame, result)
                view = await asyncio.to_thread(draw_view, result)
            except Exception as e:                      # 别让一帧异常打死整个循环
                hub.status = {"locked": False, "rows": [["错误", repr(e)]], "note": note}
                await asyncio.sleep(0.5)
                continue

            ok1, b1 = cv2.imencode(".jpg", pano, enc)
            ok2, b2 = cv2.imencode(".jpg", view, enc)
            if ok1 and ok2:
                hub.put("pano", b1.tobytes())
                hub.put("view", b2.tobytes())

            n += 1
            now = time.time()
            if now - t_prev >= 1.0:
                fps = n / (now - t_prev)
                n, t_prev = 0, now
                hub.fps = fps
            hub.status = status_payload(result, getattr(hub, "fps", 0.0),
                                        source_name, note)
            dt = time.time() - t0
            if period > 0:
                await asyncio.sleep(max(0.0, period - dt))
            else:
                await asyncio.sleep(0)


INDEX = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全景跟随 · PanoFollow</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#242c37;--fg:#e6edf3;--dim:#8b98a8;
       --accent:#5ce1e6;--ok:#3ddc84;--warn:#ffb020}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 -apple-system,"PingFang SC","Helvetica Neue",sans-serif}
 header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;
        align-items:baseline;gap:12px;flex-wrap:wrap}
 h1{font-size:16px;margin:0;letter-spacing:.3px}
 .sub{color:var(--dim);font-size:12px}
 main{padding:16px;display:grid;gap:16px;
      grid-template-columns:minmax(0,1fr) 330px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
       overflow:hidden}
 .card h2{font-size:12px;margin:0;padding:9px 12px;color:var(--dim);
          border-bottom:1px solid var(--line);font-weight:600;letter-spacing:.4px}
 .card img{display:block;width:100%;height:auto;background:#000}
 .stack{display:flex;flex-direction:column;gap:16px}
 #note{margin:0 16px 16px;padding:10px 12px;border-radius:8px;font-size:12.5px;
       background:#2a2113;border:1px solid #5c4a1d;color:var(--warn);display:none}
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 td{padding:6px 12px;border-bottom:1px solid var(--line);font-size:13px}
 td:first-child{color:var(--dim);white-space:nowrap}
 td:last-child{text-align:right}
 #badge{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line)}
 .lock{color:#0b1;border-color:#1c5}.lost{color:#f66;border-color:#633}
 footer{padding:10px 18px 26px;color:var(--dim);font-size:12px}
 code{background:#0b0e12;padding:1px 5px;border-radius:4px}
</style></head><body>
<header>
  <h1>全景跟随 · PanoFollow</h1>
  <span class="sub">X5 等距柱状全景 → 找人/跟住 → 抽出视角做动作检测 → 把信息标在平面上</span>
  <span id="badge" class="lost">连接中</span>
</header>
<div id="note"></div>
<main>
  <div class="stack">
    <div class="card"><h2>全景平面（等距柱状）· 人的方位 / 正在看的视锥 / 信息卡片</h2>
      <img src="/pano.mjpg" alt="pano"></div>
  </div>
  <div class="stack">
    <div class="card"><h2>跟随视角 · 骨架 / 脸 / 手</h2>
      <img src="/view.mjpg" alt="view"></div>
    <div class="card"><h2>这个人的信息</h2><table id="tbl"></table></div>
  </div>
</main>
<footer>第 ① 步「跟随」已修好并在实拍帧上验证。第 ② 步「贴到真实房间的地面/墙面」
  尚未做 —— 它要先估重力方向。本页的卡片标在<b>全景展开平面</b>上。</footer>
<script>
async function tick(){
  try{
    const r = await fetch('/status',{cache:'no-store'});
    const s = await r.json();
    const b = document.getElementById('badge');
    b.textContent = s.locked ? '已锁定' : '未找到人';
    b.className = s.locked ? 'lock' : 'lost';
    document.getElementById('tbl').innerHTML =
      s.rows.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
    const n = document.getElementById('note');
    if(s.note){ n.textContent = s.note; n.style.display='block'; }
    else n.style.display='none';
  }catch(e){}
}
setInterval(tick, 400); tick();
</script></body></html>
"""


async def mjpeg(request):
    from aiohttp import web
    key = request.match_info["key"]
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-store, no-cache, must-revalidate",
    })
    await resp.prepare(request)
    last = -1
    try:
        while True:
            seq, jpg = request.app["hub"].get(key)
            if jpg is None or seq == last:
                await asyncio.sleep(0.03)
                continue
            last = seq
            head = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n")
            await resp.write(head + jpg + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def index(request):
    from aiohttp import web
    return web.Response(text=INDEX, content_type="text/html", charset="utf-8")


async def status(request):
    from aiohttp import web
    return web.json_response(request.app["hub"].status)


async def snapshot(request):
    from aiohttp import web
    jpg = request.app["hub"].snapshot("pano")
    if jpg is None:
        return web.Response(status=503, text="no frame yet")
    return web.Response(body=jpg, content_type="image/jpeg")


def selftest(args):
    """不开服务，离线跑一帧并把两张图写到磁盘 —— 用来确认流水线真的通。"""
    src = cv2.imread(FALLBACK_FRAME)
    assert src is not None, f"读不到 {FALLBACK_FRAME}"
    fol = PanoFollower()
    r = fol.step(src, detect=True)
    pano = draw_pano(src, r)
    view = draw_view(r)
    cv2.imwrite("/tmp/live_selftest_pano.png", pano)
    cv2.imwrite("/tmp/live_selftest_view.png", view)
    print(f"locked={r.fix.locked} yaw={r.fix.yaw:+.2f} pitch={r.fix.pitch:+.2f} "
          f"fov={r.fix.fov:.1f} score={r.fix.score:.3f} "
          f"face={len(r.landmarks.get('face') or [])} "
          f"body={len(r.landmarks.get('body') or [])}")
    print("wrote /tmp/live_selftest_pano.png /tmp/live_selftest_view.png")
    return 0


async def serve(args):
    from aiohttp import web
    hub = Hub()
    hub.fps = 0.0
    app = web.Application()
    app["hub"] = hub
    app.router.add_get("/", index)
    app.router.add_get("/status", status)
    app.router.add_get("/snapshot.jpg", snapshot)
    app.router.add_get("/{key:pano|view}.mjpg", mjpeg)

    task = asyncio.create_task(worker(hub, args))
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f"\n  ▶  http://127.0.0.1:{args.port}\n"
          f"     全景平面 /pano.mjpg · 跟随视角 /view.mjpg · 状态 /status\n"
          f"     X5 源 {args.x5}（不出帧时自动用离线实拍帧）\n")
    try:
        await asyncio.Event().wait()
    finally:
        task.cancel()
        await runner.cleanup()


def main():
    ap = argparse.ArgumentParser(description="X5 全景跟随 + 人物信息标注 网页")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--x5", default=DEFAULT_X5, help="x5-link 服务地址")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="主循环目标帧率；<=0 表示不人为限速")
    ap.add_argument("--detect-every", type=int, default=1,
                    help="每 N 帧跑一次动作检测（holistic 是现在的性能地板）。"
                         "1=每帧都检；2 约能把帧率再翻一倍，代价是眨眼/举手这类"
                         "语义以一半频率刷新")
    ap.add_argument("--selftest", action="store_true", help="离线渲一帧自检后退出")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)
    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
