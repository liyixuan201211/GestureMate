/* GestureMate WebUI 前端
 *
 * 职责：
 *   1) getUserMedia 抓本机摄像头（服务端没有 TCC 权限，取流必须在浏览器）
 *   2) 把视频轨经 WebRTC 推给服务端（aiortc 收流后跑 mps 推理）
 *   3) 收 DataChannel 里的关键点/识别结果，在本地视频上叠骨架并更新面板
 *
 * 为什么视频不绕回来？浏览器本地就有原始画面，让服务端只回「关键点」，
 * 少一次编解码、画面最清晰、延迟最低。
 *
 * 关键点拓扑来自 /static/mp_connections.js（由 webui/gen_connections.py
 * 从 mediapipe 直接导出，跟后端同源，避免手抄索引）。
 * 必须在 app.js 之前加载 —— 它是顶层 const，不挂在 window 上。
 */

/* 兜底：万一 mp_connections.js 没加载成功，至少手部还能画 */
if (typeof HAND_CONNECTIONS === "undefined") {
  window.HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]];
}

/* 面部几个「看得出是五官」的点，用来打点强调鼻子 */
const NOSE_POINTS = [1, 6, 168, 197, 129, 358, 98, 327];

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay");
const ctx = canvas.getContext("2d");

let ws = null, pc = null, dc = null, stream = null;
let latest = null;              // 服务端最近一次 state
let lastFace = null;            // 面部点单独缓存：服务端是降频下发的
let lastFaceAt = 0;
let serverEvents = [];
let snapshotTimer = null;
let overlayRunning = false;
let configKinds = {};           // dir -> {task 类型: 个数}
let currentConfigDir = "";
/* 推理模式：browser=浏览器内跑 MediaPipe Tasks（叠加层与画面同帧，视频不上行）
              server =服务端 mps（视频上行，叠加层靠外推补偿） */
let inferMode = "browser";
let inferEngine = null;         // MediaPipe Tasks 封装（浏览器模式）
let localPose = null;           // 浏览器模式：本地推理结果（同一帧）
let inferLoopOn = false;
let lastFaceSent = 0;           // 上面部关键点的节流
let inferStats = { n: 0, ms: 0, lastMs: 0, fps: 0 };
/* 延迟补偿：把骨架上关键点按「这帧结果已经多旧」往前外推，抵消管线延迟。
   为什么默认 0：端到端延迟里，本地视频那一段（摄像头+编码+合成）和服务端
   那一段（解码+排队+推理+回传）量级接近、方向相反，会互相抵消一大部分，
   剩下的偏差我无法凭空测准 —— 所以给一个滑杆让用户看着画面 5 秒调平。
   骨架落在手后面 -> 加大；骨架跑到手前面 -> 减小。 */
let leadMs = 0;
let prevPose = null, curPose = null;   // 最近两帧关键点，用来估速度做外推

/* ------------------------------------------------------------ 连接状态 */
function setConn(text, cls) {
  $("connText").textContent = text;
  $("dot").className = "dot" + (cls ? " " + cls : "");
}

/* ------------------------------------------------------------ WebSocket */
function wsURL() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

function connectWS() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(wsURL());
    ws.onopen = () => resolve(ws);
    ws.onerror = (e) => reject(e);
    ws.onclose = () => { setConn("信令已断开", ""); };
    ws.onmessage = async (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === "answer") {
        await pc.setRemoteDescription({ type: msg.sdpType, sdp: msg.sdp });
        setConn("已连接服务端", "on");
      } else if (msg.type === "candidate") {
        try { await pc.addIceCandidate(msg.candidate); } catch (e) { /* 忽略 */ }
      } else if (msg.type === "connectionstate") {
        if (msg.state === "connected") setConn("WebRTC 已连通", "on");
        else if (msg.state === "connecting") setConn("正在建立连接…", "wait");
        else if (["failed", "closed", "disconnected"].includes(msg.state))
          setConn("连接中断：" + msg.state, "");
      }
    };
  });
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function wsSendOrDC(obj) {
  if (dc && dc.readyState === "open") dc.send(JSON.stringify(obj));
  else wsSend(obj);
}

/* ------------------------------------------------------------ 摄像头 */
async function listCameras() {
  const sel = $("camSelect");
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter(d => d.kind === "videoinput");
    sel.innerHTML = "";
    cams.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `摄像头 ${i + 1}`;
      sel.appendChild(o);
    });
    const fe = cams.find(c => /facetime|built-?in|内建|内置/i.test(c.label));
    if (fe) sel.value = fe.deviceId;
    return cams;
  } catch (e) {
    return [];
  }
}

async function getCamera() {
  const [w, h] = $("resSelect").value.split("x").map(Number);
  const deviceId = $("camSelect").value;
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30 },
      ...(deviceId ? { deviceId: { exact: deviceId } } : {})
    }
  });
}

/* ------------------------------------------------------------ 启动/停止 */
async function start() {
  $("btnStart").disabled = true;
  setConn("请求摄像头权限…", "wait");
  try {
    stream = await getCamera();
  } catch (e) {
    const name = e && e.name || "Error";
    let tip = e.message || String(e);
    if (name === "NotAllowedError")
      tip = "摄像头被拒绝。请在浏览器地址栏左侧的权限图标里允许摄像头，然后重试。";
    else if (name === "NotFoundError") tip = "没有找到可用的摄像头设备。";
    else if (name === "NotReadableError") tip = "摄像头被其它程序占用（比如另一个 App 正在用它）。";
    $("placeholder").textContent = "打不开摄像头：" + tip;
    $("placeholder").style.display = "flex";
    setConn("摄像头不可用", "");
    $("btnStart").disabled = false;
    return;
  }

  video.srcObject = stream;
  $("placeholder").style.display = "none";
  await listCameras();                    // 授权后 label 才有值

  // 先问服务端：推理在哪跑？这决定了要不要把视频推上去
  try {
    const st = await (await fetch("/api/status")).json();
    inferMode = (st.server && st.server.infer) || "browser";
  } catch (e) { inferMode = "browser"; }
  document.body.dataset.infer = inferMode;
  // 服务端视角只有服务端推理时才存在；浏览器模式改用「姿态模型」这一档
  $("serverViewRow").style.display = inferMode === "browser" ? "none" : "";
  $("complexityRow").style.display = inferMode === "browser" ? "none" : "";
  $("poseModelRow").style.display = inferMode === "browser" ? "" : "none";

  // 浏览器模式：先把模型加载好（要下 ~9MB wasm + 三个模型，首帧还要编译着色器）
  if (inferMode === "browser") {
    setConn("正在加载识别模型…", "wait");
    $("placeholder").textContent = "正在加载 MediaPipe 模型（首次约几秒）…";
    $("placeholder").style.display = "flex";
    try {
      const { createEngine } = await import("/static/infer.js");
      inferEngine = await createEngine({
        poseModel: $("poseModel").value || "lite",
        delegate: "GPU",
        log: (m) => console.log("[infer]", m),
      });
    } catch (e) {
      $("placeholder").textContent = "模型加载失败：" + (e.message || e);
      setConn("模型加载失败", "");
      $("btnStart").disabled = false;
      return;
    }
    $("placeholder").style.display = "none";
  }

  setConn("正在建立 WebRTC…", "wait");
  await connectWS();

  pc = new RTCPeerConnection({ iceServers: [] });   // 本机/局域网，host 候选足够
  if (inferMode === "server") {
    stream.getVideoTracks().forEach(t => {
      // 告诉编码器这是「运动」内容：优先保帧率、少缓冲 -> 端到端延迟更低
      try { t.contentHint = "motion"; } catch (e) { /* 忽略 */ }
      pc.addTrack(t, stream);
    });
    // 低延迟优先：宁可掉画质，也不要让编码器排队积压（排队 = 纯延迟）
    const vs = pc.getSenders().find(x => x.track && x.track.kind === "video");
    if (vs) {
      try {
        const p = vs.getParameters();
        if (!p.encodings || !p.encodings.length) p.encodings = [{}];
        p.encodings[0].maxBitrate = 2500000;      // 720p30 够用
        p.degradationPreference = "realtime";
        await vs.setParameters(p);
      } catch (e) { /* 个别浏览器不支持，忽略 */ }
    }
  }
  // 浏览器模式：**不加视频轨** —— 关键点走 DataChannel，视频一步都不出浏览器

  dc = pc.createDataChannel("state");
  dc.onopen = () => {
    setConn("数据通道已就绪", "on");
    syncFaceSetting();                    // 把当前的面部下发开关同步给服务端
  };
  dc.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    handleServer(msg);
  };
  dc.onclose = () => setConn("数据通道已关闭", "");

  pc.onicecandidate = (e) => {
    if (e.candidate)
      wsSend({ type: "candidate", candidate: {
        candidate: e.candidate.candidate,
        sdpMid: e.candidate.sdpMid,
        sdpMLineIndex: e.candidate.sdpMLineIndex } });
  };
  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    if (s === "connected") setConn("WebRTC 已连通", "on");
    else if (s === "connecting") setConn("正在建立连接…", "wait");
    else if (["failed", "disconnected", "closed"].includes(s)) setConn("连接中断：" + s, "");
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  wsSend({ type: "offer", sdp: offer.sdp, sdpType: offer.type });

  $("btnStop").disabled = false;
  if (!overlayRunning) { overlayRunning = true; requestAnimationFrame(drawSkeleton); }
  if (inferMode === "browser") startInferLoop();
}

/* ------------------------------------------------- 浏览器内推理（逐帧） */
/* 用 requestVideoFrameCallback：每来一个**真正的新视频帧**才跑一次推理，
   拿到 metadata 还能知道这帧的时间。这样叠加层画的永远是刚算过的那一帧 ——
   对齐是构造上成立的，不需要外推预测。
   没有这个 API 就退回 rAF（会重复算同一帧，但至少能跑）。 */
function startInferLoop() {
  if (inferLoopOn || !inferEngine) return;
  inferLoopOn = true;
  inferStats = { n: 0, ms: 0, lastMs: 0, fps: 0 };
  const hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  const step = () => {
    if (!inferLoopOn) return;
    runInferOnce();
    if (hasRVFC) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  if (hasRVFC) video.requestVideoFrameCallback(step);
  else requestAnimationFrame(step);
}

function runInferOnce() {
  if (!inferEngine || !video.videoWidth) return;
  const ts = performance.now();          // 只需单调递增
  let r;
  const t0 = performance.now();
  try {
    r = inferEngine.process(video, ts);
  } catch (e) {
    console.warn("推理失败", e);
    return;
  }
  const cost = performance.now() - t0;
  localPose = { face: r.face, body: r.body, hands: {
    leftHand: r.leftHand, rightHand: r.rightHand } };
  // 统计（给面板显示浏览器侧开销）
  inferStats.n++;
  const now = performance.now();
  inferStats.cost = inferStats.cost ? inferStats.cost * 0.9 + cost * 0.1 : cost;
  if (inferStats.lastMs) {
    const gap = now - inferStats.lastMs;
    inferStats.ms = inferStats.ms ? inferStats.ms * 0.9 + gap * 0.1 : gap;
    if (inferStats.n % 30 === 0) inferStats.fps = 1000 / inferStats.ms;
  }
  inferStats.lastMs = now;

  sendLandmarks(r);
}

/* 关键点上行给服务端：它只负责跑任务系统 + 语义分析。
   身体/手每帧都发（很小）；脸 468 点体积大，按 8Hz 节流，关掉「画面部」就完全不发。 */
function sendLandmarks(r) {
  if (!dc || dc.readyState !== "open") return;
  const now = Date.now();
  const wantFace = $("drawFace").checked;
  const sendFace = wantFace && (now - lastFaceSent >= 125);
  if (sendFace) lastFaceSent = now;
  try {
    dc.send(JSON.stringify({
      type: "landmarks",
      frameT: now / 1000,
      face: sendFace ? r.face : null,
      body: r.body,
      leftHand: r.leftHand,
      rightHand: r.rightHand,
    }));
  } catch (e) { /* 通道刚关掉，忽略 */ }
}

function stop() {
  inferLoopOn = false;
  if (inferEngine) { try { inferEngine.close(); } catch (e) {} inferEngine = null; }
  localPose = null;
  if (pc) { try { pc.close(); } catch {} pc = null; }
  if (ws) { try { ws.close(); } catch {} ws = null; }
  dc = null;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  latest = null; lastFace = null;
  $("placeholder").textContent = "已停止。点击「开始」重新授权摄像头。";
  $("placeholder").style.display = "flex";
  $("btnStart").disabled = false;
  $("btnStop").disabled = true;
  setConn("未连接", "");
}

/* ------------------------------------------------------------ 服务端消息 */
function handleServer(msg) {
  if (msg.type === "state") {
    const s = msg.state;
    latest = s;
    // 留最近两帧带时间戳的关键点，供延迟补偿估速度用
    if (s.frameT && (s.hands || s.body)) {
      prevPose = curPose;
      curPose = { t: s.frameT, body: s.body, hands: s.hands };
    }
    // 面部是降频下发的：只在真有数据时更新缓存，否则overlay 会 5Hz 闪烁
    if (s.face) { lastFace = s.face; lastFaceAt = performance.now(); }
    updateMetrics(s);
    updateFacePanel(s);
    updateBodyPanel(s);
    if (msg.events && msg.events.length) appendEvents(msg.events);
  } else if (msg.type === "hello") {
    console.log("server hello", msg);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* 管线没在跑 / 配置加载失败时，必须一眼看见 ——
   否则用户看到的是「画面在动、所有数字不动」，只能猜。 */
function updateBanner(s) {
  const el = $("banner");
  const msgs = [];
  let err = false;
  if (s.configError) {
    err = true;
    msgs.push(`<b>任务配置加载失败</b>，已保留原来的配置（识别不受影响）。`);
    msgs.push(`原因：<code>${escapeHtml(s.configError)}</code>`);
    msgs.push(`常见情况是配置里引用的文件找不到（例如 match 的 poseFile 是按相对路径写的）。`
      + `换一个任务配置，或把对应文件补齐即可。`);
  }
  if (s.pipelineAlive === false) {
    err = true;
    msgs.push(`<b>识别管线没有在运行</b>，所以下面的数字不会变化。`);
  }
  if (!msgs.length) { el.style.display = "none"; return; }
  el.className = "banner" + (err ? " err" : "");
  el.innerHTML = msgs.join("<br>");
  el.style.display = "block";
}

function updateMetrics(s) {
  if (!s) return;
  updateBanner(s);
  $("mFps").textContent = s.fps ? s.fps.toFixed(1) : "–";
  if (inferMode === "browser") {
    // 浏览器模式：推理在本地，耗时自己量；服务端只报「收到间隔」和「处理耗时」
    $("mInfer").textContent = inferStats.cost
      ? inferStats.cost.toFixed(1) + " ms" : "–";
    $("mLoop").textContent = (s.loopMs != null)
      ? s.loopMs.toFixed(2) + " ms" : "–";
  } else {
    $("mInfer").textContent = s.inferMs ? s.inferMs.toFixed(1) + " ms" : "–";
    $("mLoop").textContent = s.loopMs ? s.loopMs.toFixed(1) + " ms" : "–";
  }
  $("mFrames").textContent = s.frames ?? "–";
  $("mDropped").textContent = s.dropped ?? 0;
  // 「结果从服务端算完到发到浏览器」的真实耗时（服务端侧延迟）
  $("mLatency").textContent = (s.latencyMs != null)
    ? s.latencyMs.toFixed(1) + " ms" : "–";
  $("mEngine").textContent = s.engine || "–";
  $("mSource").textContent = "帧源：" + (s.source || "–") +
    "；任务配置：" + (s.dataDir ? String(s.dataDir).split("/").filter(Boolean).pop()
                                : "（无，只做识别）") +
    (s.error ? "  ⚠️ " + s.error : "");
  $("partsHint").textContent = "引擎需要的部位：" + (s.parts || []).join(", ");

  const hands = s.hands || {};
  for (const [key, id] of [["leftHand", "handLeft"], ["rightHand", "handRight"]]) {
    const el = $(id), pts = hands[key];
    el.classList.toggle("on", !!pts);
    el.querySelector(".pill").textContent = pts ? `已检出 ${pts.length} 点` : "未检出";
  }

  const tl = $("taskList"), tasks = s.tasks || {};
  const ids = Object.keys(tasks);
  if (!ids.length) {
    tl.innerHTML = '<li class="empty">（未加载配置：只做识别，不跑任务）</li>';
  } else {
    tl.innerHTML = ids.map(id => {
      const t = tasks[id];
      return `<li><span>${id}</span><span class="tag">${t.type} ·
        <b class="${t.active ? "t-on" : "t-off"}">${t.active ? "激活" : "待命"}</b></span></li>`;
    }).join("");
  }
}

function setBar(el, ratio, color) {
  if (!el) return;
  const pct = Math.max(0, Math.min(1, ratio || 0)) * 100;
  el.style.width = pct.toFixed(1) + "%";
  if (color) el.style.background = color;
}

function updateFacePanel(s) {
  const f = s && s.features;
  const hasFace = !!(f && f.available);
  if (!hasFace) {
    $("fEye").textContent = "未检出";
    $("fBlink").textContent = f ? f.blinkCount : "–";
    $("fMouth").textContent = "未检出";
    $("fHead").textContent = "未检出";
    $("fPose").textContent = "–";
    setBar($("fEyeRight"), 0, "#5ce1e6");
    setBar($("fEyeLeft"), 0, "#5ce1e6");
    setBar($("fMouthBar"), 0, "#ff6b81");
    $("fNote").textContent = $("drawFace").checked
      ? "未检测到人脸（把脸放进画面里）"
      : "已关闭面部下发（勾上「画面部」恢复）";
    return;
  }

  const e = f.eyes || {};
  const rClosed = !!e.rightClosed, lClosed = !!e.leftClosed;
  $("fEye").textContent = (rClosed || lClosed)
    ? `闭眼（右${rClosed ? "闭" : "睁"} / 左${lClosed ? "闭" : "睁"}）`
    : "睁开";
  setBar($("fEyeRight"), e.rightOpen, rClosed ? "#ffb020" : "#5ce1e6");
  setBar($("fEyeLeft"), e.leftOpen, lClosed ? "#ffb020" : "#5ce1e6");
  $("fBlink").textContent = `${f.blinkCount} 次`;

  const m = f.mouth || {};
  $("fMouth").textContent = m.state
    ? `${m.state}${m.open != null ? `（MAR ${m.open}）` : ""}`
    : "未检出";
  setBar($("fMouthBar"), (m.open || 0) / 0.4, "#ff6b81");

  const h = f.head;
  $("fHead").textContent = h ? h.direction : "–";
  $("fPose").textContent = h
    ? `${h.yaw} / ${h.pitch} / ${h.rollDeg}°`
    : "–";
  $("fNote").textContent = `面部 ${f.landmarkCount || 0} 点 · 朝向为粗略估计` +
    (lastFace ? " · 面部 5Hz 下发" : "");
}

function updateBodyPanel(s) {
  const f = s && s.features;
  const b = f && f.body;
  if (!b) {
    $("bCount").textContent = "未检出";
    $("bHandsUp").textContent = "–";
    $("bTilt").textContent = "–";
    return;
  }
  $("bCount").textContent = `${b.landmarkCount} / 33`;
  const hu = b.handsUp || {};
  $("bHandsUp").textContent = (hu.left || hu.right)
    ? `举手：${hu.left ? "左" : ""}${hu.right ? "右" : ""}`
    : "双手未举";
  $("bTilt").textContent = `${b.shoulderTiltDeg}°（${b.lean}）`;
}

function appendEvents(events) {
  const ul = $("eventList");
  for (const e of events) {
    const key = e.t + "|" + e.text;
    if (serverEvents.includes(key)) continue;
    serverEvents.push(key);
    const li = document.createElement("li");
    li.className = e.kind === "warn" ? "warn" : "";
    const d = new Date(e.t * 1000);
    li.innerHTML = `<span>${e.text}</span><span class="tag">${
      String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${
      String(d.getSeconds()).padStart(2, "0")}</span>`;
    ul.insertBefore(li, ul.firstChild);
    while (ul.children.length > 40) ul.removeChild(ul.lastChild);
  }
  const empty = ul.querySelector(".empty");
  if (empty) empty.remove();
}

/* ------------------------------------------------------------ 骨架叠加 */
function px(pts, w, h) {
  return (i) => [pts[i][0] * w, pts[i][1] * h];
}

function strokeConn(ctx2, at, conns, color, lw) {
  ctx2.strokeStyle = color;
  ctx2.lineWidth = lw;
  ctx2.beginPath();
  for (const [a, b] of conns) {
    const p = at(a), q = at(b);
    ctx2.moveTo(p[0], p[1]);
    ctx2.lineTo(q[0], q[1]);
  }
  ctx2.stroke();
}

function dotList(ctx2, at, idxs, color, r) {
  ctx2.fillStyle = color;
  for (const i of idxs) {
    const p = at(i);
    ctx2.beginPath();
    ctx2.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx2.fill();
  }
}

/* ---- 延迟补偿：按时间外推关键点 ---- */
function extrapolate(cur, prev, f) {
  if (!cur || !prev || cur.length !== prev.length) return cur;
  const out = new Array(cur.length);
  for (let i = 0; i < cur.length; i++) {
    const p = prev[i], c = cur[i];
    out[i] = [c[0] + (c[0] - p[0]) * f,
              c[1] + (c[1] - p[1]) * f,
              c[2] + (c[2] - p[2]) * f];
  }
  return out;
}

/* 叠加层用哪份关键点。
   浏览器模式：本地推理结果 —— 就是**正在显示的这一帧**算出来的，同帧、零延迟，
               不需要任何外推。
   服务端模式：服务端回传的关键点，天然落后一两帧，用外推补偿。 */
function currentPose() {
  if (inferMode === "browser") return localPose;
  return predictedPose();
}

function predictedPose() {
  if (!curPose) return null;  // 结果已经多旧了：frameT 来自服务端 time.time()，同一台机器上可直接比
  let ageMs = (Date.now() / 1000 - curPose.t) * 1000;
  if (!isFinite(ageMs)) ageMs = 0;
  ageMs = Math.max(0, Math.min(500, ageMs));   // 跨机器有钟差时兜一下
  const lead = ageMs + leadMs;
  if (!prevPose || lead <= 0.5) return curPose;
  const dt = (curPose.t - prevPose.t) * 1000;
  if (dt <= 1 || dt > 500) return curPose;
  // 倍数上限 2.5：手突然停住时纯线性外推会“冲过头”，必须夹住
  const f = Math.min(lead / dt, 2.5);
  return {
    body: extrapolate(curPose.body, prevPose.body, f),
    hands: {
      leftHand: extrapolate(curPose.hands && curPose.hands.leftHand,
                            prevPose.hands && prevPose.hands.leftHand, f),
      rightHand: extrapolate(curPose.hands && curPose.hands.rightHand,
                             prevPose.hands && prevPose.hands.rightHand, f),
    },
  };
}

function drawSkeleton() {
  const w = video.videoWidth, h = video.videoHeight;
  if (w && h) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);

    const lw = Math.max(1.2, w / 640);
    const r = Math.max(1.6, w / 380);
    const pose = currentPose();       // 浏览器模式=同帧结果；服务端模式=外推补偿

    /* ---- 面部（先画，压在身体/手下面） */
    // 浏览器模式用本地同帧结果；服务端模式用回传的（降频下发，超时就别画了）
    const facePts = inferMode === "browser"
      ? (pose && pose.face)
      : ((lastFace && performance.now() - lastFaceAt < 1500) ? lastFace : null);
    if ($("drawFace").checked && facePts) {
      const at = px(facePts, w, h);
      if ($("drawMesh").checked && typeof FACE_TESSELATION !== "undefined") {
        strokeConn(ctx, at, FACE_TESSELATION, "rgba(255,255,255,0.10)", lw * 0.7);
      }
      if (typeof FACE_CONTOURS !== "undefined")
        strokeConn(ctx, at, FACE_CONTOURS, "rgba(226,236,255,0.55)", lw * 0.9);
      if (typeof FACE_LEFT_EYEBROW !== "undefined")
        strokeConn(ctx, at, FACE_LEFT_EYEBROW, "#ffd166", lw * 1.2);
      if (typeof FACE_RIGHT_EYEBROW !== "undefined")
        strokeConn(ctx, at, FACE_RIGHT_EYEBROW, "#ffd166", lw * 1.2);
      if (typeof FACE_LEFT_EYE !== "undefined")
        strokeConn(ctx, at, FACE_LEFT_EYE, "#5ce1e6", lw * 1.3);
      if (typeof FACE_RIGHT_EYE !== "undefined")
        strokeConn(ctx, at, FACE_RIGHT_EYE, "#5ce1e6", lw * 1.3);
      if (typeof FACE_LIPS !== "undefined")
        strokeConn(ctx, at, FACE_LIPS, "#ff6b81", lw * 1.3);
      dotList(ctx, at, NOSE_POINTS, "#ffa94d", r * 1.1);
      dotList(ctx, at, [1], "#ff7a1a", r * 1.8);              // 鼻尖
      dotList(ctx, at, [33, 133, 362, 263], "#5ce1e6", r);    // 眼角
      dotList(ctx, at, [13, 14, 61, 291], "#ff6b81", r);      // 唇
    }

    /* ---- 身体 */
    if ($("drawBody").checked && pose && pose.body &&
        typeof POSE_CONNECTIONS !== "undefined") {
      const at = px(pose.body, w, h);
      strokeConn(ctx, at, POSE_CONNECTIONS, "#a78bfa", lw * 1.6);
      dotList(ctx, at, [...Array(33).keys()], "#c4b5fd", r);
      dotList(ctx, at, [11, 12, 13, 14, 15, 16], "#f0abfc", r * 1.5);
    }

    /* ---- 手（画在最上层） */
    if ($("drawHands").checked && pose && pose.hands) {
      for (const [key, color] of [["leftHand", "#3ddc84"], ["rightHand", "#4ea1ff"]]) {
        const pts = pose.hands[key];
        if (!pts) continue;
        const at = px(pts, w, h);
        strokeConn(ctx, at, HAND_CONNECTIONS, color, lw * 1.5);
        dotList(ctx, at, [...Array(pts.length).keys()], color, r);
      }
    }
  }
  requestAnimationFrame(drawSkeleton);
}

/* ------------------------------------------------------------ 控件 */
function applyMirror() {
  const on = $("mirror").checked;
  video.classList.toggle("mirror", on);
  canvas.classList.toggle("mirror", on);
}

function syncFaceSetting() {
  wsSendOrDC({ type: "setFace", enabled: $("drawFace").checked, rate: 5 });
}

async function loadConfigs() {
  try {
    const r = await fetch("/api/configs");
    const { configs } = await r.json();
    const sel = $("configSelect");
    sel.innerHTML = '<option value="">（只做识别，不跑任务）</option>';
    for (const c of configs) {
      const o = document.createElement("option");
      o.value = c.dir;
      const kinds = Object.entries(c.tasks || {}).map(([k, v]) => `${k}×${v}`).join(" ");
      o.textContent = `${c.name}  ${kinds}`;
      sel.appendChild(o);
      configKinds[c.dir] = c.tasks || {};
    }
  } catch (e) { /* ignore */ }
}

function pollSnapshot(on) {
  if (snapshotTimer) { clearInterval(snapshotTimer); snapshotTimer = null; }
  $("serverViewCard").style.display = on ? "block" : "none";
  if (!on) return;
  const img = $("snapshot");
  const tick = () => { img.src = "/api/snapshot?t=" + Date.now(); };
  tick();
  snapshotTimer = setInterval(tick, 650);
}

function bindUI() {
  $("btnStart").onclick = start;
  $("btnStop").onclick = stop;

  $("camSelect").onchange = async () => {
    if (!stream) return;
    try {
      const s = await getCamera();
      stream.getTracks().forEach(t => t.stop());
      stream = s;
      video.srcObject = s;
      const sender = pc && pc.getSenders().find(x => x.track && x.track.kind === "video");
      if (sender) await sender.replaceTrack(s.getVideoTracks()[0]);
    } catch (e) { console.warn("切换摄像头失败", e); }
  };

  $("resSelect").onchange = () => $("camSelect").onchange();
  // 浏览器模式下换姿态模型：重建引擎、重开推理循环（不用重连 WebRTC）
  $("poseModel").onchange = async () => {
    if (inferMode !== "browser" || !inferEngine) return;
    inferLoopOn = false;
    setConn("正在切换姿态模型…", "wait");
    try { inferEngine.close(); } catch (e) {}
    try {
      const { createEngine } = await import("/static/infer.js");
      inferEngine = await createEngine({
        poseModel: $("poseModel").value, delegate: "GPU",
        log: (m) => console.log("[infer]", m),
      });
      startInferLoop();
      setConn("数据通道已就绪", "on");
    } catch (e) {
      setConn("模型切换失败：" + (e.message || e), "");
    }
  };
  $("complexity").onchange = (e) =>
    wsSendOrDC({ type: "setComplexity", value: Number(e.target.value) });
  // 有些官方 demo 会**真的操作电脑**（command 跑命令、keypress 模拟按键）。
  // 切换前确认一次 —— 否则用户以为只是「换个识别配置」，
  // 结果焦点窗口被自动打字，这个提示比事后解释便宜得多。
  $("configSelect").onchange = (e) => {
    const dir = e.target.value || null;
    const kinds = configKinds[dir] || {};
    const hit = ["command", "keypress", "request", "socketsend"].filter(k => kinds[k]);
    if (hit.length) {
      const desc = hit.map(k => `${k}×${kinds[k]}`).join("、");
      const name = dir.split("/").filter(Boolean).pop();
      const ok = confirm(
        `这个配置包含会操作电脑的任务：${desc}\n\n` +
        `· command：运行系统命令（例如启动计算器）\n` +
        `· keypress：模拟按键，会往当前焦点窗口输入\n\n` +
        `确认加载「${name}」吗？`);
      if (!ok) {
        e.target.value = currentConfigDir;
        return;
      }
    }
    currentConfigDir = dir || "";
    wsSendOrDC({ type: "setConfig", dataDir: dir });
  };

  $("mirror").onchange = applyMirror;
  $("drawFace").onchange = syncFaceSetting;
  $("showServerView").onchange = (e) => {
    pollSnapshot(e.target.checked);
    // 服务端只在需要「服务端视角」时才画骨架 —— 否则那 3.2ms/帧纯属白烧
    wsSendOrDC({ type: "setServerView", enabled: e.target.checked });
  };
  $("leadMs").oninput = (e) => {
    leadMs = Number(e.target.value) || 0;
    $("leadMsVal").textContent = leadMs + " ms";
  };
}

window.addEventListener("load", async () => {
  bindUI();
  applyMirror();
  await listCameras();
  await loadConfigs();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("placeholder").textContent =
      "这个浏览器不支持 getUserMedia。请用 http://127.0.0.1 打开（localhost 才算安全上下文）。";
    setConn("浏览器不支持", "");
  }
});
