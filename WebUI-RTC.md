# GestureMate WebUI（WebRTC）说明

把 GestureMate 的手势识别做成网页：**浏览器取本机摄像头 → WebRTC 送上服务端 →
服务端跑已经优化过的 mps 管线 → 关键点与任务状态经 DataChannel 回到网页**。

启动：

```bash
./run_webui.sh                 # 默认浏览器取流
# 浏览器打开 http://127.0.0.1:8770 ，点「开始」并在弹窗里「允许」摄像头
```

---

## 一、为什么摄像头放在浏览器侧取？

这是本次实现最关键的一个决定，来自实测：

| 尝试 | 结果 |
|---|---|
| 服务端 `cv2.VideoCapture(0)` | ❌ `OpenCV: not authorized to capture video (status 0)` |
| 服务端 `ffmpeg -f avfoundation -i 1` | ❌ 协商完像素格式后**卡住，随后被 macOS SIGKILL** |
| 浏览器 `getUserMedia` | ✅ 弹窗点「允许」即可 |

原因：dsh 是以 **launchd 用户代理**（`launchctl list` 里的 `ai.deepseek.dsh.web`，
父进程是 launchd）运行的，进程树没有 GUI App 身份，macOS 的 TCC 无法给它弹窗授权，
于是直接拒绝／杀掉。浏览器是正规 GUI App，弹窗一点就行。

**结论**：取流放浏览器，推理留 Python —— 两边各做各擅长的，而且推理仍然是那套
我们优化过的 `LandmarkEngine` + `TaskController`，识别结果与命令行版完全一致。

> 如果以后给终端（或这个服务进程）授了摄像头权限，用
> `./run_webui.sh --source opencv` 或 `--source ffmpeg` 就能切回服务端取流，
> 前端代码一行都不用改。

---

## 二、拓扑

```
┌──────────── 浏览器 ────────────┐        ┌──────────── 服务端 (Python) ────────────┐
│                                │        │                                        │
│  getUserMedia(内置摄像头)       │        │   aiohttp: 静态页 + /ws 信令            │
│        │                       │        │        │                               │
│        ├──► <video> 本地直接显示│        │        └──► aiortc RTCPeerConnection    │
│        │      (不绕服务端，      │  RTP   │                 │                      │
│        │       画面最清晰)       ├───────►│  收到视频轨 ──► BrowserPushSource      │
│        │                       │ 视频上行│                 │                      │
│  canvas 叠加骨架 ◄──────────────┼────────┤  GesturePipeline 线程：                │
│        ▲                       │DataChnl│   镜像 → LandmarkEngine(holistic)      │
│        └── 关键点 JSON ─────────┼────────┤   → extractLandmarks → TaskController  │
│                                │        │   → 状态/事件发布                       │
└────────────────────────────────┘        └────────────────────────────────────────┘
```

* **视频只上行**。服务端不回视频：少一次编解码，浏览器用本地原始画面叠骨架，
  画质与延迟都最好。
* 信令用 WebSocket（WebRTC 必需），媒体走 SRTP，**同一条 PeerConnection**。
* 纯本机/局域网，host candidate 足够，默认不配 STUN（离线也能用）。
* 服务端丢帧策略是**丢旧留新**（`queue(maxsize=1)`），宁可丢帧也不让延迟堆积。

### 坐标约定（易错点）

命令行版推理前会水平镜像画面，再在 `extractLandmarks` 里左右手互换来抵消镜像。
WebUI 沿用同一套语义，但为了前端不用再猜，`pipeline._unmirror()` 会把关键点 x
还原到**相机原始（未镜像）坐标系**再发出；网页端用 CSS `scaleX(-1)` 同时镜像
`<video>` 和 `<canvas>`，两者始终对齐。

---

## 三、功能

### 3.1 识别能力（手 / 面部 / 身体）

服务端跑的是 holistic，本来就同时产出三组关键点；WebUI 现在把它们全部画出来，
并在服务端算了**语义结论**（`webui/face_body.py`，纯函数、可单测）：

| 部位 | 画出来的 | 算出来的语义 |
|---|---|---|
| 面部 468 点 | 面部轮廓、**眉毛**、**眼睛**、**嘴唇**、**鼻子**（不同颜色分组） | 眼睛开合度 EAR（右/左）、**眨眼计数**、嘴巴开合度 MAR + 状态（闭合/微张/张开）、头部朝向（yaw/pitch/roll + 正对/朝左/朝右/抬头/低头） |
| 身体 33 点 | 全身骨架（肩/肘/腕/髋/膝/踝） | 关键点覆盖数 n/33、**举手**（腕高于肩）、肩线倾角 + 左倾/右倾/正 |
| 手 21×2 | 左右手骨架（绿=左，蓝=右） | 由原有 Task 系统消费（detect/match/keypress/…） |

叠层可分别开关：**画手 / 画身体 / 画面部 / 面部网格**（最后一项是完整的
2556 条三角网格，比较费，默认关）。

### 3.2 关键点怎么传

* 拓扑表由 `webui/gen_connections.py` 从 mediapipe **直接导出**成
  `static/mp_connections.js`（手 21、身体 35、面部轮廓 124、网格 2556 条）。
  不手抄索引 —— 抄错一两个数字图还是画得出来，只是连错线，很难发现。
* 身体 33 点很小，每帧（10Hz）都发。
* **面部 468 点体积大**：默认降到 **5Hz** 下发（前端用缓存渲染，不会闪），
  并支持前端用 `setFace` 关掉或调频（取消勾选「画面部」即停止下发）。
* 坐标统一还原到**相机原图坐标系**再发，前端直接画在本地视频上；
  网页用 CSS `scaleX(-1)` 同时镜像 video 与 canvas，两者始终对齐。

### 3.3 其他

* 指标面板：服务端 FPS / 推理 ms / 单帧 ms / 已处理帧数 / 丢帧 / 引擎与部位
* 任务列表（id/类型/激活与否）、**触发事件流**
* 控件：摄像头选择（优先内置 FaceTime）、分辨率、**模型复杂度**、
  任务配置切换、镜像、三种叠层开关、**服务端视角**
* 端点：`/api/status`、`/api/snapshot`（带骨架的 JPEG）、`/api/configs`

---

## 四、验证结果

### 1) 无头 RTC 客户端（`tests/rtc_client.py`）

它用 aiortc 把「浏览器那一半」演出来（喂测试视频当摄像头），
可以完整验证服务端 RTC 通路：

```
连接状态        : connected（曾经 connected: True）
ICE 状态        : completed
DataChannel     : open
已发送视频帧    : 109
收到 state 消息 : 79
服务端已处理帧  : 213
服务端 FPS      : 13.8
服务端推理      : 27.92 ms / 单帧 35.01 ms
引擎            : holistic  parts=['leftHand', 'rightHand']
手部            : 左 ✗ 右 ✓
任务            : ['left', 'right']
触发事件        : ['right', 'right', 'right', 'right', ...]
管线报错        : None
✅ RTC 通路验证通过：信令/ICE/DTLS/视频上行/状态回传全部工作
```

### 2) 文件帧源（无需摄像头）

```bash
./.venv/bin/python webui/server.py --source file \
    --video tests/fixtures/hands_gestures.mp4 --data tests/_e2e_data --port 8770
curl -s localhost:8770/api/status   # frames 持续增长, error: null
curl -s localhost:8770/api/snapshot -o /tmp/s.jpg   # 带骨架的 JPEG
```

实测 27.8 FPS、推理 27.86 ms、单帧 35.03 ms，任务 `left`/`right` 均触发。

另外两个不用浏览器的检查：

```bash
# 面部/身体语义逻辑（合成关键点，秒级）
.venv/bin/python tests/test_face_body.py

# 延迟补偿外推逻辑（从 app.js 抠真函数来测）
node tests/test_extrapolation.mjs

# 延迟剖析：分段耗时；加 --sweep 看输入尺寸×复杂度
.venv/bin/python tests/bench_latency.py
.venv/bin/python tests/bench_latency.py --sweep

# 无头 RTC 端到端（信令/ICE/DTLS/上行/回传 + 状态更新率 + 复杂度切换回归）
.venv/bin/python tests/rtc_client.py --fps 30 --seconds 8 --toggle-complexity

# 关键点拓扑表有变动时重新生成
.venv/bin/python webui/gen_connections.py
```

### 3) 真实浏览器（ego lite / Chromium，内置摄像头）

在真实浏览器里跑通，实测数据：

```
连接状态        : 数据通道已就绪
摄像头          : FaceTime HD Camera (2C0E:82E3)   ← 电脑自带摄像头
video / canvas  : 1280x720 / 1280x720（正在播放）
服务端 FPS      : 29.9
推理 / 单帧     : 27.9 ms / 32.1 ms
已处理          : 221 帧（丢帧 3）
引擎            : holistic   parts=leftHand,rightHand
任务            : left / right (Detect · 激活)
触发事件        : right ×N
```

即：**浏览器取内置摄像头 → WebRTC 上行 → 服务端 mps 推理 30 FPS →
关键点与任务状态回传 → 骨架叠加与事件面板正常**，整条链路闭环。

（一次性完整跑完 633 帧，仅丢 3 帧 —— 丢旧留新策略生效，没有延迟堆积。）

### 4) 面部 / 身体识别（真实浏览器 + 内置摄像头）

```
连接状态        : 数据通道已就绪
摄像头          : FaceTime HD Camera (2C0E:82E3)
服务端 FPS      : 27.4    推理 28.6 ms / 单帧 34.5 ms
面部识别        : 眼睛 睁开（右/左开合度条满）  眨眼 16 次
                  嘴巴 微张（MAR 0.08）
                  头部朝向 正对   yaw/pitch/roll = -0.021 / 0.897 / 7.2°
                  面部 468 点
身体            : 关键点 33/33   举手 双手未举   肩线 -2.9°（正）
手部            : 左 21 点 / 右 21 点
```

画面上：面部轮廓 + 眉毛（黄）+ 眼睛（青）+ 嘴唇（粉）+ 鼻子（橙）逐一对齐，
身体骨架从肩到腿完整贴合，双手 21 点分别用绿/蓝画出。

另外 `test_face_body.py` 用**合成关键点**覆盖了 7 项语义逻辑
（左右眼按 x 判定、roll 归一化、yaw 符号、单帧眨眼、嘴巴分级、举手、空输入安全）。

---

## 五、文件

| 文件 | 作用 |
|---|---|
| `webui/pipeline.py` | 帧源抽象（browser/opencv/ffmpeg/file）+ mps 推理线程 + 状态与事件发布 + 坐标还原 |
| `webui/face_body.py` | 面部/身体语义分析（EAR 眼睛、MAR 嘴巴、头朝向、眨眼、举手、肩线） |
| `webui/server.py` | aiohttp（静态页/信令/状态接口）+ aiortc（收流、DataChannel）+ 看门狗 |
| `webui/gen_connections.py` | 从 mediapipe 导出关键点拓扑表到前端 JS（避免手抄索引） |
| `webui/static/mp_connections.js` | 自动生成：手/身体/面部轮廓/面部网格等连接表 |
| `webui/static/index.html` | 页面结构 |
| `webui/static/app.js` | getUserMedia、RTCPeerConnection、手/脸/身体叠层绘制、面板更新 |
| `webui/static/style.css` | 样式 |
| `run_webui.sh` | 一键启动（会自动补装 aiortc/aiohttp） |
| `tests/rtc_client.py` | 无头 RTC 客户端，端到端验证服务端（含 `--toggle-complexity`） |
| `tests/bench_latency.py` | 延迟剖析：分段耗时 / 输入尺寸×复杂度扫描 |
| `webui/static/infer.js` | 浏览器端推理封装（MediaPipe Tasks，ES module） |
| `webui/static/mediapipe/` | Tasks 运行时与模型（45MB，见 9.7） |
| `tests/test_landmark_engine.py` | 关键点模式：语义+任务链、`parse_landmarks` 脏输入、坏配置 |
| `tests/test_extrapolation.mjs` | 延迟补偿外推逻辑的回归测试（从 app.js 抠出真函数 + 固定时钟） |
| `tests/test_face_body.py` | 面部/身体语义的合成关键点单测（7 项） |
| `tests/bench_mps.py` / `bench_engine.py` | mps 管线/引擎基准（见《部署与优化.md》） |

新增依赖：`aiortc`（WebRTC）、`aiohttp`（HTTP+WebSocket）。其余复用 GestureMate
原有依赖与代码，`Main.py` 命令行版**未受影响**。

---

## 六、踩坑记录（都已在代码里修掉）

### 1. `self._stop` 把 `threading.Thread._stop()` 覆盖了 ⚠️ 最坑的一个

管线里原本写 `self._stop = threading.Event()` 作为停止标志。看着没问题，
但 `threading.Thread` **内部就有一个 `_stop()` 方法**，`join()` 会去调它：

```
TypeError: 'Event' object is not callable
  File "threading.py", line 1171, in _wait_for_tstate_lock
    self._stop()
```

后果特别隐蔽：**只有在「重启管线」时才炸**，也就是你一动 UI 上的
「模型复杂度」或「任务配置」就会触发。炸了之后旧管线停了、新管线没起来，
表现就是「画面在动，但所有数字都不动」——非常容易误判成前端问题。
现已改名 `_stopEvent`，并给 `restart_pipeline` 加了 try/except，
再加一个**看门狗**：每秒检查管线线程，发现死了就自动拉起。

回归测试：`tests/rtc_client.py --toggle-complexity`（会连续切 2→0→1
并断言每次切换后 frames 仍在增长）。

### 2. 负步长视图不能直接给 OpenCV 画

`frame[:, ::-1, :]` 是**不连续**的负步长视图，OpenCV 绘图函数写不进去：

```
error: (-5:Bad argument) in function 'line'
> Layout of the output array img is incompatible with cv::Mat
```

命令行版之所以没暴露，是因为它先 `cv2.cvtColor` 生成了新的连续数组再画。
现改为显式 `np.ascontiguousarray(...)`，并让「画骨架失败」不再拖垮整条管线。

### 3. 逐帧事件把服务端日志刷爆

有手在画面里时，`processing left/right` 每秒几十行，日志完全没法看。
现用 `EventMarker` + `DropWebuiEvents` 两个 filter：事件照旧喂给网页，
但不再往控制台打。

### 4. 帧源生命周期不能挂在管线上

管线重启会走 `finally`，如果在那里 `source.close()`，
`opencv`/`file` 帧源一关就废，重启后拿到的是死句柄。
现在帧源归服务端管，只在服务退出时关闭。

### 5. FaceMesh 的左右手性按「画面哪一边」定，不按解剖学 ⚠️

做面部识别时踩到的。实测同一帧：

```
喂原图        : idx33 x=0.485（画面左）   idx263 x=0.519（画面右）
喂镜像图      : idx33 x=0.481（画面左）   idx263 x=0.514（画面右）   ← 几乎没变
镜像图再 unmirror: idx33 x=0.519（画面右）  idx263 x=0.486（画面左）
```

也就是说，把图镜像之后 FaceMesh 仍把 idx33 放在**那张图的左侧**，
并不跟着解剖学换到另一只眼。我们的管线是「镜像推理 + 坐标 unmirror」，
于是交付坐标里 idx33 反而跑到了画面右侧 —— 如果按「33 就是右眼」写死，
右眼/左眼的读数会整个反过来。

修法：**一律按交付坐标里 x 的大小判左右**（画面左 = 被摄者右眼），
完全绕开 MediaPipe 的编号习惯（`face_body.eye_groups()`）。
（顺带一提：手部模型不是这样 —— 它用分类器判手性，镜像后会翻，
所以原代码那个 left↔right 互换是必须保留的。）

### 6. roll 必须归一化到 (-90, 90]

因为上面第 5 条，`atan2` 的两个端点取反了，`rollDeg` 直接读出 **-178.1°** ——
一个「头正对着镜头」的人显示成几乎倒过来。归一到 (-90, 90] 之后，
正脸稳定在 0° 附近（实测 113 帧：中位 -0.30°，范围 -32~19°）。

### 7. 眨眼判定不能用平滑后的 EAR，也不能按眼计数

两个坑叠在一起，症状是「一直眨眼但计数始终 0」或「眨一次跳 2」：

* **不能用 EMA 平滑值判阈值**：平滑会把单帧眨眼抹掉 —— 实测原始 EAR 最低到
  0.23，平滑后最低只到 0.34，永远碰不到 `ear_closed`。现在平滑值只用于显示
  「开合度」，睁/闭判定走原始值。
* **不能每只眼各自 +1**：双眼同时眨，一次眨眼会被数成 2 次。
  现在两只眼的「闭→睁」跳变合并成一次计数。

修完实测：真实浏览器里对着摄像头说话，`眨眼 16 次`，符合预期。

### 8. 配置相对路径 + 重启顺序 + 失控重试：一次「卡死」的完整成因 ⚠️

**症状**：在界面上把「任务配置」切成计算器 demo（`example/data_example`）之后，
画面还在动（浏览器本地视频照常播放），但**所有数字都不再变化**，
看起来就是「卡死」。

三个问题叠在一起，各修一处：

**① 配置里的相对路径（真正的根因）**
`match` 的 `poseFile` 写的是 `"./hand/hand0.json"` —— 相对**配置所在目录**。
命令行版靠 `os.chdir(dataDir)` 才能找到；WebUI 不能 chdir（那是全局进程状态），
于是 `MatchTask.__init__` 直接：

```
FileNotFoundError: [Errno 2] No such file or directory: './hand/hand0.json'
```

修法：`TaskController.readConfig(path, base_dir=None)` 显式接收基准目录
（默认取 `path` 所在目录，所以命令行版行为不变），把相对 `poseFile` 解析成绝对路径。

**② 我自己的重启顺序错了**
原来的 `restart_pipeline` 是「先 `old.stop()`，再构造新管线」。构造一抛异常，
旧管线已经死了、新的又没起来，`self.pipeline` 还指着那具尸体
—— 这就是「画面在动、数字不动」的直接原因。
改成**先构造成功、再换掉旧的**：新配置有问题时，旧管线毫发无伤，识别不受影响。

**③ 看门狗失控重试**
加了自愈之后，它每秒重试一次、每次都失败，**刷了几千行日志、持续好几分钟**，
而且永远起不来。改成：连续失败 3 次就停止重试，把原因留在 `configError` 里。

**另外补的三件事**（都是「让问题可见」，比修 bug 本身更重要）：
* `configError` 随状态下发，界面顶部弹红色告警条，写明失败原因和建议；
* `pipelineAlive` 也下发 —— 界面能区分「在跑但很慢」和「根本没在跑」；
* 任务配置下拉在切到含 `command`/`keypress` 的配置前**弹确认框**
  （这些任务会真的运行命令、模拟按键往焦点窗口打字，不说清楚很危险）。

顺带修了一个误导性显示：任务清单原来只在帧循环里更新，
没接视频流时界面一直显示「未加载配置」，像是加载失败；现在启动即发布。

回归测试：`tests/test_fixes.py` 的 PASS 8（相对 poseFile 解析）与 PASS 9（未知 task type 报错）。

---

## 七、延迟优化（「几乎零延迟」能做到什么程度）

### 7.1 先把时间拆开（`tests/bench_latency.py`）

| 阶段 | 中位 | 占比 |
|---|---|---|
| **推理 mps** | **28.5 ms** | **78%** |
| 镜像 + 转 RGB | 4.24 ms | 12% |
| 服务端绘制骨架 | 3.21 ms | 9% |
| 提取关键点 | 0.19 ms | 0.5% |
| 坐标还原 | 0.35 ms | 1% |
| 语义分析（脸/身体） | 0.03 ms | 0.1% |
| **合计** | **36.50 ms** | → 27.4 FPS |

结论：**瓶颈就是推理，别的都是零头**。我加的面部/身体那套处理一共只花 0.6ms。

### 7.2 推理本身砍不动了（都实测过）

* **降分辨率没用**：1280×720 与 480×360 的推理耗时几乎一样（28.5 vs 28.4ms，
  像素差 6.4 倍）。因为 MediaPipe 内部会把输入缩放到各模型固定的输入尺寸，
  喂小图不省算力。（我之前 `--proc-size=640` 测不出收益，就是这个原因。）
* **各模型单独开销**（50 帧实测）：

  | 模型 | 中位 |
  |---|---|
  | pose c=1 | 15.62 ms ← 最大头（占 holistic 的 56%） |
  | pose c=0 | 9.62 ms |
  | face_mesh | **1.62 ms**（几乎免费） |
  | hands 单独（开跟踪） | 34.86 ms |
  | hands 单独（每帧检测） | 24.49 ms |

  holistic 的 28ms ≈ pose 15.6 + 双手（借 pose 推出的 ROI，很便宜）10.8 + 脸 1.6。
  单跑 hands 反而要 25~35ms，所以 **holistic 已经是最省的那条路**，拆开只会更慢。
* 因此推理下限就是 **~22ms（c=0）/ ~28ms（c=1）**。c=0 能省 6ms，但手部检出率
  从 82% 掉到 68%（pose 精度下降 -> 手部 ROI 变差），所以只作为可选项。

### 7.3 真正省下来的是这三处

**① 固定 10Hz 推送 → 事件驱动（省 ~19ms 延迟 + 3 倍刷新率）**

原来推送循环是 `while: send(); await sleep(0.1)`，而管线每 ~36ms 就出一个新结果。
两边节奏不一致，实测（仿真 + 真机）：

| | 前端更新率 | 结果平均年龄 | 最大 |
|---|---|---|---|
| 固定 10Hz 轮询 | 9.7 /s | 18.8 ms | 37.1 ms |
| **事件驱动** | **27.7 /s** | **0.7 ms** | 1.9 ms |

注意：**单纯把轮询提到 30Hz 没用**（仍积压 17.9ms），必须由「有新结果」触发。
这不只是延迟问题 —— 视频 30fps 而叠加层只有 10Hz，看起来就是「骨架跟不上手」。

**② 镜像改用 `cv2.flip`（省 3.6ms，零精度损失）**

`f[:, ::-1, :]` 是负步长视图，`np.ascontiguousarray` 得逐元素按步长拷：
实测 **3.93ms**；`cv2.flip` 是 SIMD 优化的，只要 **0.39ms**。
一行之差，白省 3.5ms。（顺带：负步长视图还不能直接给 OpenCV 绘图函数写。）

**③ 服务端不再画骨架（省 3.2ms）**

浏览器本来就自己画叠加层，服务端那份只服务于「服务端视角」。
改成默认不画、勾选该开关时才画（运行时可切，不用重启管线）。

**合计：单帧 36.50ms → 29.53ms（→ 33.9 FPS，+24%）**，
再加结果延迟 18.8ms → 0.1ms。真机实测（内置摄像头）：

```
状态更新率      : 30.1 /s      （以前固定推送封顶 ~10/s）
服务端结果延迟  : 0.1 ms       （以前平均白等 19ms、最高 37ms）
客户端结果年龄  : 0.39 ms      （前端 + 同一台机器时钟）
FPS            : 31.3          （以前 27~30）
```

### 7.4 「完全零延迟」不可能，原因在物理

端到端延迟 = 摄像头曝光读出（~30-60ms）+ 浏览器编码（~10-30ms）+ 传输 +
排队 + **推理 28ms**（不可再降）+ 回传。**摄像头那一段就注定有几十毫秒**，
服务器再快也消不掉。

另外还有一个容易忽略的事实：浏览器本地显示视频本身也有 30-70ms 的延迟，
而它和服务端那一段（解码+排队+推理+回传）量级接近、方向相反，
**会互相抵消一大部分** —— 所以叠加层并不像「两段相加」那么滞后。

### 7.5 最后一点偏差：用「延迟补偿」调平

剩下那点偏差取决于摄像头/编码/合成的具体延迟，我在服务端测不到，
所以给了一个滑杆而不是猜一个默认值：

**延迟补偿 0~150ms**（控件在页面底部）。骨架落在手后面就加大，
骨架跑到手前面就减小。原理是把关键点按「这帧结果已经多旧」（`frameT` 时间戳，
同机时钟可直接比）加上你给的补偿量做**线性外推**，并夹住倍数上限防止手停住时冲过头。
默认 0（因为 7.4 那条抵消效应，默认往往已经基本对齐）。

外推的确切公式（`app.js: predictedPose`）：

```
lead = clamp(now - frameT, 0, 500ms) + 你给的补偿ms
f    = min(lead / dt, 2.5)              # dt = 相邻两帧结果的间隔；2.5 是硬上限
点   = 当前点 + (当前点 - 上一帧点) * f
```

注意「当前位置本身就含一帧位移」，所以 f=2.5 时总位移是 `1 + 2.5 = 3.5` 帧的距离，
而不是 2.5 帧 —— 我第一次手算校验时就在这里算错了。

这段逻辑默认配置下**根本不会执行**（lead ≤ 0.5 就直接返回当前帧），
所以专门加了 `tests/test_extrapolation.mjs`：从 `app.js` 里把这两个函数抠出来、
注入固定时钟来测（真实时钟会让「造数据到调用之间流逝的几毫秒」混进来，
第一版测试就是这么假失败的），覆盖 10 项：分段位移、夹断上限、年龄计入、
无前一帧、dt=0、点数不一致、未检出的手保持 null。

### 7.6 顺手核实的几件事（都不是瓶颈，但得确认）

* **前端 canvas 绘制完全不是瓶颈**：面部网格那 2556 条线段，实测只占 0.11ms/帧；
  把网格开关打开，rAF 帧率从 29.9 掉到 29.7（噪声级）。所以没做「分层缓存」那种优化。
* **服务端绘制开关有效**：勾上「服务端视角」单帧 +1.8~3.2ms，关掉即恢复 —— 与 7.3③ 的测算一致。
* **WebRTC 低延迟参数**（`contentHint="motion"`、`degradationPreference="realtime"`、
  `maxBitrate=2.5Mbps`）加上后视频正常、无 JS 报错。

---

## 八、已知限制

1. **摄像头授权需要你点一下**：浏览器首次会弹「是否允许使用摄像头」。
2. **`--infer server` 时只认一个「帧提供者」**：多个浏览器接入时，只有第一个推流的那台
   上报数据，其余页面能看状态但不驱动推理（避免互相打架）。断开会自动让位给下一个。
   （`--infer browser` 模式下每个页面各自本地推理，不存在这个问题。）
3. `--infer server` 时服务端不做视频回转，网页上看到的是**浏览器本地画面 +
   服务端算出的骨架**；想看服务端到底喂了什么给 mps，开「服务端视角」。
   浏览器模式下服务端没有画面，该开关已隐藏。
4. 仅在 `http://127.0.0.1` 或 `https` 下可用（`getUserMedia` 要求安全上下文）。
   想在手机上用，走 `http://<局域网IP>:8770` 时浏览器可能仍要求 https。

---

## 九、浏览器内推理模式（`--infer browser`，当前默认）

### 9.1 为什么把推理搬进浏览器

第七章算过：能省的延迟都省掉之后，剩下的大头是**摄像头曝光读出 + 浏览器编码**，
服务器再快也消不掉。根因在架构 —— 视频要绕服务端一圈，叠加层用的必然是
**一两帧之前**的结果。

把推理放进浏览器后，叠加层就能画在**它自己刚算过的那一帧**上，
对齐是构造上成立的，不是靠 7.5 那种外推预测。

### 9.2 拓扑（视频一步都不出浏览器）

    浏览器  --getUserMedia-->  <video> ──► MediaPipe Tasks ──► 画骨架（同一帧）
                                        └── 关键点 --DataChannel--> 服务端
    服务端  --DataChannel-->  语义特征 + 任务事件  -->  浏览器

* 服务端仍然负责它不可替代的部分：**TaskController 任务系统 + face_body 语义**
  （单一实现，不用把 EAR/MAR 那套数学在 JS 里再抄一遍）。
* Python 侧的 `TaskController` / `face_body.py` / 任务配置**一行没改**，
  因为浏览器算出的关键点格式与 `extractLandmarks` 的输出完全一致。

### 9.3 实测（Apple M2 Max，Chrome 内核）

| 项目 | 数据 |
|---|---|
| 建引擎 | 143 ms |
| 首帧（含着色器编译） | 539 ms（**冷启动首次曾达 11s**，故必须预热） |
| 稳态推理（1280×720，GPU） | **21.3 ms → 46.9 FPS** |
| 分段 | 脸 4.0 / 身 9.7 / 手 7.7 ms |
| 稳态推理（CPU） | 45.2 ms（**GPU 快 2.1 倍**，故默认 GPU 并保留 CPU 回退） |
| pose full vs lite（GPU） | 21.7 vs 21.3 ms —— 几乎无差，可放心选 full |
| **服务端单帧开销** | **0.05~0.16 ms**（原 mps 管线 29.5ms，**轻约 600 倍**） |
| 浏览器端 JS 开销 | 24.4 ms/帧 |
| `pc.getSenders()` | **`[]`** —— 视频轨根本没建立 |

浏览器端推理（21.3ms）甚至比服务端 mps（28ms）更快，而且没有编码/传输/排队。

### 9.4 手性（handedness）—— 以真人实测为准，**不互换**

MediaPipe 文档说手性判定「假定输入图已镜像（自拍）」，非镜像输入请自行互换。
但**照那条做是错的**：真人举左手，骨架画在手上、颜色却是蓝的（`rightHand`）。
实测结论是 —— 对**未镜像的原图**，MediaPipe 直接给出的就是物理正确的手性，
`SWAP_HANDEDNESS = false`。

⚠️ **我在这一条上先犯过错，过程值得记下来**：当时我拿 fixture 图对照
「浏览器（原图）vs 旧管线（镜像+swap）」，算出中点 x 分别是 0.566 / 0.567、
标签都是 `rightHand`，就下了「完全一致、可以放心」的结论。

问题在于：那只证明两个实现**一致**，不证明它们**正确**。
参考基准（旧管线）本身就是错的，验证再自洽也没用。
真正的判据只有**真人举手**这一条 —— 而这正是我做不到、必须请你实测的那一步。

**副作用（要留意）**：旧管线 `Utils.extractLandmarks` 是把左右手对调的，
所以它录制的姿态文件（`example/data_example` 里 `GetPoseJson` 生成的那批）
标签是**反的**。用新语义跑那些配置时，需要用另一只手触发，或者重新录一遍。
（录制和播放都用同一套语义，所以旧 demo 自洽，只是文件里的标签与物理左右相反。）

### 9.5 关键点上行

* 身体 / 双手**每帧都发**（很小）；脸 468 点**按 8Hz 节流**，取消勾选「画面部」就完全不发。
* 上行前把坐标压到 4 位小数（脸那 468 个点不压会白白多一半流量）。
* 服务端收到后必须过 `parse_landmarks()` 校验：截断点数、丢非法项、夹到 [0,1]、挡 NaN/Inf。
  浏览器算「半个信任边界」，一个 NaN 就能顺任务链把整条路搞崩。

### 9.6 两个模式怎么选

    ./run_webui.sh                          # 默认 --infer browser
    ./run_webui.sh --infer server --source file \
        --video tests/fixtures/hands_gestures.mp4   # 旧的服务端 mps 那条路

* 前端会先问 `/api/status` 拿模式，再决定**要不要把视频轨加进去**；
  浏览器模式下不加轨（`pcSenders == []`），服务端也就不会走 `_consume`。
* UI 随模式切换：浏览器模式隐藏「模型复杂度 / 服务端视角」，改显示「姿态模型」。
* `--infer server` 那条路完整保留，也仍是命令行版（`Main.py`）用的实现。

### 9.7 代价

1. **模型资源 45 MB**（`webui/static/mediapipe/`：wasm 9.5MB + nosimd 9.4MB +
   face 3.8MB + pose_lite 5.8MB + pose_full 9.4MB + hand 7.8MB）。首次加载要下载。
   * 建议：浏览器端只用 **nosimd 或 simd 其中之一**即可省 ~9MB（取决于目标机器）。
2. **冷启动首帧可能十几秒**（WebGL 着色器编译）。需要预热并给出加载态，
   否则用户会以为卡死。目前的做法是先建引擎再开推理循环，并显示「正在加载模型…」。
3. 浏览器需支持 WebGL/WASM（现代 Chrome/Edge/Safari 均可；WebGPU 版更快但尚未通用）。
4. 服务端「服务端视角 / 快照」在这个模式下没有意义，已隐藏/404。

### 9.8 这个模式的测试

| 测试 | 覆盖 |
|---|---|
| `tests/test_landmark_engine.py` | 真实检出灌进引擎：语义特征 + 任务链 + 单帧开销；`parse_landmarks` 的四种脏输入；无配置；坏配置不影响识别 |
| `tests/test_extrapolation.mjs` | 服务端模式那套外推补偿（浏览器模式用不到，但保留） |
| 浏览器实测 | 引擎加载、无视频轨、关键点到服务端、UI 模式切换（见 9.3） |
