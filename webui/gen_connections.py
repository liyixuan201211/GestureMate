#!/usr/bin/env python3
"""把 MediaPipe 的连接表（骨架/轮廓）导出成前端用的 JS 常量。

为什么要生成而不是手写：关键点索引（比如「左眼是 362/385/387/263/373/380」）
抄错一两个数字很难发现，图还能画出来、只是连错线。直接从 mediapipe 里读，
保证前后端用的是同一套拓扑。

    .venv/bin/python webui/gen_connections.py
    # 生成 webui/static/mp_connections.js
"""
import os
import sys

import mediapipe.python.solutions as sol

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "static", "mp_connections.js")

SETS = [
    ("HAND_CONNECTIONS", sol.hands.HAND_CONNECTIONS),
    ("POSE_CONNECTIONS", sol.pose.POSE_CONNECTIONS),
    ("FACE_CONTOURS", sol.face_mesh.FACEMESH_CONTOURS),
    ("FACE_TESSELATION", sol.face_mesh.FACEMESH_TESSELATION),
    ("FACE_LEFT_EYE", sol.face_mesh.FACEMESH_LEFT_EYE),
    ("FACE_RIGHT_EYE", sol.face_mesh.FACEMESH_RIGHT_EYE),
    ("FACE_LEFT_EYEBROW", sol.face_mesh.FACEMESH_LEFT_EYEBROW),
    ("FACE_RIGHT_EYEBROW", sol.face_mesh.FACEMESH_RIGHT_EYEBROW),
    ("FACE_LIPS", sol.face_mesh.FACEMESH_LIPS),
    ("FACE_FACE_OVAL", sol.face_mesh.FACEMESH_FACE_OVAL),
    ("FACE_NOSE", sol.face_mesh.FACEMESH_NOSE),
]


def main():
    lines = [
        "/* 由 webui/gen_connections.py 自动生成，请勿手改。",
        " * 数据来自 mediapipe.python.solutions，与后端同源。",
        " */",
    ]
    for name, conns in SETS:
        pairs = sorted(tuple(sorted(p)) for p in conns)
        body = ",".join(f"[{a},{b}]" for a, b in pairs)
        lines.append(f"const {name} = [{body}];")
        print(f"{name:22s} {len(pairs):5d} 条")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写入 {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())
