#!/usr/bin/env node
/* 「延迟补偿」外推逻辑的回归测试。
 *
 *   node tests/test_extrapolation.mjs
 *
 * 为什么单独测它：这段数学很微妙（我自己第一次手算就错了 —— 忘了当前位置
 * 本身就含一帧位移）。它又只在滑杆 > 0 时才真正执行，默认配置下跑不到，
 * 坏了也不容易被发现。
 *
 * 做法：直接从 webui/static/app.js 里把两个函数**抠出来**求值，
 * 保证测的就是线上那份代码，而不是抄一份副本。
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = readFileSync(join(ROOT, "webui/static/app.js"), "utf8");

/** 按名字取出一段完整的 function（用大括号配平，避免正则截断） */
function grab(name) {
  const start = SRC.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`app.js 里找不到 function ${name}`);
  let depth = 0, i = SRC.indexOf("{", start);
  for (let k = i; k < SRC.length; k++) {
    if (SRC[k] === "{") depth++;
    else if (SRC[k] === "}") {
      depth--;
      if (depth === 0) return SRC.slice(start, k + 1);
    }
  }
  throw new Error(`${name} 的大括号不配平`);
}

const CODE = grab("extrapolate") + "\n" + grab("predictedPose") + "\n";

/* 固定时钟。
   注意 predictedPose 会把「这帧结果已经多旧」算进 lead（ageMs），
   如果用真实 Date.now()，从造数据到调用之间流逝的那几毫秒会混进来，
   测出来的位移就带着 ±0.0015 的抖动（第一版测试就是这么假失败的）。
   所以这里把 Date 当参数注进去，让时间完全可控。 */
const NOW_MS = 1_700_000_000_000;

/** 用给定的 (curPose, prevPose, leadMs) 造一个隔离的求值环境 */
function make(cur, prev, lead, nowMs = NOW_MS) {
  const f = new Function(
    "curPose", "prevPose", "leadMs", "Date",
    CODE + "\nreturn { extrapolate, predictedPose };");
  return f(cur, prev, lead, { now: () => nowMs });
}

const hand = (shift = 0) =>
  Array.from({ length: 21 }, (_, i) =>
    [0.30 + (i % 5) * 0.02 + shift, 0.60 + Math.floor(i / 5) * 0.03, 0]);
const body = (shift = 0) =>
  Array.from({ length: 33 }, (_, i) =>
    [0.20 + (i % 6) * 0.06 + shift, 0.40 + Math.floor(i / 6) * 0.05, 0]);

/** dt: 这两帧相隔多少毫秒；shift: 后一帧相对前一帧移了多少（模拟运动） */
function poses(dt, shift, ageMs = 0) {
  const now = NOW_MS / 1000;
  return {
    cur: { t: now - ageMs / 1000, body: body(shift),
           hands: { leftHand: hand(shift), rightHand: null } },
    prev: { t: now - ageMs / 1000 - dt / 1000, body: body(0),
            hands: { leftHand: hand(0), rightHand: null } },
  };
}

let pass = 0, fail = 0;
/* 容差 1e-6：这些位移是「归一化画面宽度的比例」，1e-6 相当于 1280 像素里
   的 0.001 像素，完全无意义。之所以不能要求位级相等，是因为 ageMs 要经过
   「epoch 秒相减」（1.7e9 量级的双精度只剩 ~1e-7 相对精度）。
   管线本身也把坐标四舍五入到 4 位小数。 */
const TOL = 1e-6;
function check(name, actual, expect) {
  const ok = Math.abs(actual - expect) < TOL;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}: ${actual}（期望 ${expect}）`);
  ok ? pass++ : fail++;
}

// 一帧位移 0.01、间隔 40ms、结果年龄 0
const F = 0.01;
const p = poses(40, F);

// lead=0：不加外推，就是当前帧
check("lead=0   位移", make(p.cur, p.prev, 0).predictedPose().hands.leftHand[0][0] - 0.30, F);
// lead=20：0.01 + 0.01*(20/40)
check("lead=20  位移", make(p.cur, p.prev, 20).predictedPose().hands.leftHand[0][0] - 0.30, F + F * 0.5);
// lead=60：0.01 + 0.01*(60/40)
check("lead=60  位移", make(p.cur, p.prev, 60).predictedPose().hands.leftHand[0][0] - 0.30, F + F * 1.5);
// lead=600：倍数被夹在 2.5 -> 0.01 + 0.01*2.5（防止手停住时无限冲出去）
check("lead=600 被夹在 2.5 倍", make(p.cur, p.prev, 600).predictedPose().hands.leftHand[0][0] - 0.30, F + F * 2.5);

// 结果本身已经旧了：年龄要算进 lead
const old = poses(40, F, 60);          // 这帧是 60ms 前算出来的
check("年龄计入 lead", make(old.cur, old.prev, 0).predictedPose().hands.leftHand[0][0] - 0.30, F + F * 1.5);

// 边界与健壮性
{
  const f = make(p.cur, null, 60);     // 只有一帧，没有速度可估
  const r = f.predictedPose();
  check("没有前一帧时退回当前帧", r.hands.leftHand[0][0] - 0.30, F);

  const same = poses(0, F);            // dt=0，不能除零
  const r2 = make(same.cur, same.prev, 60).predictedPose();
  check("dt=0 不炸且退回当前帧", r2.hands.leftHand[0][0] - 0.30, F);

  const bad = make(p.cur, { t: p.prev.t, body: [1, 2], hands: null }, 60);
  const r3 = bad.predictedPose();
  console.log(`  ${r3 && r3.body.length === 33 ? "PASS" : "FAIL"}  点数不一致时退回当前帧（body=${r3.body.length}）`);
  r3 && r3.body.length === 33 ? pass++ : fail++;

  const r4 = make(p.cur, p.prev, 60).predictedPose();
  console.log(`  ${r4.hands.rightHand === null ? "PASS" : "FAIL"}  未检出的那只手保持 null（不会造出假点）`);
  r4.hands.rightHand === null ? pass++ : fail++;
  console.log(`  ${r4.body.length === 33 && r4.hands.leftHand.length === 21 ? "PASS" : "FAIL"}  点数保持 33 / 21`);
  r4.body.length === 33 && r4.hands.leftHand.length === 21 ? pass++ : fail++;
}

console.log(`\n${fail === 0 ? "全部通过。" : `有 ${fail} 项失败。`}（${pass} passed, ${fail} failed）`);
process.exit(fail === 0 ? 0 : 1);
