/* 浏览器端推理封装（MediaPipe Tasks Vision）。
 *
 * 为什么把推理搬到浏览器：视频要绕服务端一圈，叠加层用的必然是一两帧之前的
 * 结果；在浏览器里算，叠加层就能画在**它自己刚算过的那一帧**上，对齐是构造上
 * 成立的，不靠外推预测。
 *
 * 输出格式与 Python 侧保持一致（都是归一化 [x,y,z]，坐标系=相机原图）：
 *     { face: [[x,y,z] x468] | null,
 *       body: [[x,y,z] x33]  | null,
 *       bodyWorld: [[x,y,z] x33] | null,   // 米制 3D 世界坐标，给动作识别算膝角
 *       leftHand: [...]|null, rightHand: [...]|null }
 * 这样服务端的 face_body.py 和 TaskController 可以一行不改地复用。
 *
 * 注意：这里**不镜像**。getUserMedia 给的就是未镜像的原图，直接喂进去、
 * 直接按原图坐标画，再用 CSS 把 video 和 canvas 一起镜像出「自拍感」。
 * （旧管线是「镜像推理 + 坐标还原」，绕了一圈；现在不需要了。）
 *
 * ⚠️ 上面这条"不镜像"直接决定动作识别的左右语义：坐标是**相机原图**，
 *    所以画面 +x 是玩家的**左边** → `LandmarkTaskEngine` 里那个
 *    `ActionDetector(mirrored_input=False)`。命令行版反而是 True
 *    （TaskController 先镜像画面再推理）。这正是 §5.7 说最容易写反的地方。
 */

const MP_DIR = "/static/mediapipe";
const WASM_DIR = `${MP_DIR}/wasm`;
const MODEL_DIR = `${MP_DIR}/models`;

/** handedness 映射。
 *
 * **实测结论：不互换。**
 * MediaPipe 文档说手性判定「假定输入图已镜像（自拍）」，非镜像输入请自行互换 ——
 * 但按那条做，真人举左手会被标成 rightHand（实测：骨架画在手上、颜色却是蓝的）。
 * 也就是说：对**未镜像的原图**，MediaPipe 直接给出的就是**物理正确**的手性，
 * 不需要换。（文档那句的适用方向和我原先理解相反。）
 *
 * 教训：当时我拿 fixture 图对照「浏览器(原图) vs 旧管线(镜像+swap)」得到
 * 「完全一致」就下了结论 —— 但那只证明两个实现**一致**，不证明它们**正确**。
 * 参考基准本身是错的，验证再自洽也没用。真正的判据只能是真人举手的实测。
 *
 * 注意副作用：旧管线(`Utils.extractLandmarks`)是把左右手对调的，所以它录制的
 * 姿态文件（如 `example/data_example` 里 `GetPoseJson` 生成的那批）标签是反的。
 * 用新语义跑那些配置时，需要用另一只手触发，或者重新录一遍。
 */
const SWAP_HANDEDNESS = false;

function toXYZ(list) {
  if (!list || !list.length) return null;
  const out = new Array(list.length);
  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    // 压到 4 位小数：脸有 468 个点，不压会让上行流量白白大一圈
    out[i] = [Math.round(p.x * 1e4) / 1e4,
              Math.round(p.y * 1e4) / 1e4,
              Math.round((p.z || 0) * 1e4) / 1e4];
  }
  return out;
}

export async function createEngine({
  poseModel = "lite",        // lite | full
  delegate = "GPU",          // GPU | CPU
  wantFace = true, wantPose = true, wantHands = true,
  log = () => {},
} = {}) {
  const mod = await import(`${MP_DIR}/vision_bundle.mjs`);
  const { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } = mod;
  const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);

  const base = (modelAssetPath) => ({ modelAssetPath, delegate });

  async function make(kind, Ctor, modelPath, extra) {
    const opts = {
      baseOptions: base(`${MODEL_DIR}/${modelPath}`),
      runningMode: "VIDEO",
      ...extra,
    };
    try {
      return await Ctor.createFromOptions(fileset, opts);
    } catch (e) {
      // GPU delegate 在个别环境下会失败，退回 CPU（慢一些但能用）
      if (opts.baseOptions.delegate === "GPU") {
        log(`${kind} 的 GPU 后端不可用，退回 CPU：${e.message || e}`);
        opts.baseOptions.delegate = "CPU";
        return await Ctor.createFromOptions(fileset, opts);
      }
      throw e;
    }
  }

  const face = wantFace ? await make("face", FaceLandmarker, "face_landmarker.task", {
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  }) : null;

  const pose = wantPose ? await make("pose", PoseLandmarker,
    poseModel === "full" ? "pose_landmarker_full.task" : "pose_landmarker_lite.task", {
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
      // 世界坐标（米制 3D）。动作识别的膝角判据必须用它：
      // 正面机位下蹲时髋-膝-踝在画面里近乎共线，2D 投影膝角恒 ≈180°，
      // **根本判不出蹲**。见《向星而行-UE实现设计.md》§5.1 与 action/features.py。
      outputWorldLandmarks: true,
    }) : null;

  const hands = wantHands ? await make("hands", HandLandmarker, "hand_landmarker.task", {
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  }) : null;

  log(`引擎就绪：pose=${poseModel} delegate=${delegate} `
      + `face=${!!face} pose=${!!pose} hands=${!!hands}`);

  return {
    /** 对一帧做推理。tsMs 必须单调递增。 */
    process(source, tsMs) {
      const out = { face: null, body: null, bodyWorld: null,
                    leftHand: null, rightHand: null };
      let t0 = performance.now();
      if (face) {
        const r = face.detectForVideo(source, tsMs);
        if (r && r.faceLandmarks && r.faceLandmarks.length)
          out.face = toXYZ(r.faceLandmarks[0]);
      }
      const tFace = performance.now() - t0;
      t0 = performance.now();
      if (pose) {
        const r = pose.detectForVideo(source, tsMs);
        if (r && r.landmarks && r.landmarks.length)
          out.body = toXYZ(r.landmarks[0]);
        // 世界坐标（米制）。缺了它动作识别就只能靠 2D 膝角 —— 正面机位下
        // 那个量恒 ≈180°，蹲下判不出来（见文件顶部与 outputWorldLandmarks）。
        if (r && r.worldLandmarks && r.worldLandmarks.length)
          out.bodyWorld = toXYZ(r.worldLandmarks[0]);
      }
      const tPose = performance.now() - t0;
      t0 = performance.now();
      if (hands) {
        const r = hands.detectForVideo(source, tsMs);
        const lms = (r && r.landmarks) || [];
        const hd = (r && r.handednesses) || r && r.handedness || [];
        for (let i = 0; i < lms.length; i++) {
          const cat = (hd[i] && hd[i][0] && hd[i][0].categoryName) || "";
          let side = cat === "Left" ? "leftHand" : (cat === "Right" ? "rightHand" : null);
          if (!side) continue;
          if (SWAP_HANDEDNESS) side = side === "leftHand" ? "rightHand" : "leftHand";
          out[side] = toXYZ(lms[i]);
        }
      }
      const tHands = performance.now() - t0;
      out.timing = { face: tFace, pose: tPose, hands: tHands };
      return out;
    },
    close() {
      try { face && face.close(); } catch (e) {}
      try { pose && pose.close(); } catch (e) {}
      try { hands && hands.close(); } catch (e) {}
    },
  };
}
