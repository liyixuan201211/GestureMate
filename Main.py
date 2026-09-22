import os
import sys
import getopt
import logging

def showHelp():
    print("Usage:")
    print("\tMain.py -h | --help")
    print(
        "\tMain.py [--fps=<fps>] [--complexity=0|1|2] [--data=\"<path to data>\"] [--cvshow]"
    )
    print(
        "\t        [--video=<file>] [--proc-size=<px>] [--log-level=info|debug|warning]"
    )
    print("")
    print("Options:")
    print("\t-h --help\t\tshow this help message and exit")
    print(
        "\t--fps=<fps>\t\ttarget fps when tracking (fps>=0, 0 means unlimited) [default: 8]"
    )
    print(
        "\t--complexity=0|1|2\ttrack model complexity (0 for lite, 1 for medium, 2 for heavy)"
        " [default: 1]"
    )
    print(
        "\t--data=\"<path to data>\"\tuse config file in folder <path to data> [default: \"./data\"]"
    )
    print(
        "\t--cvshow\t\tuse opencv to display camera view"
    )
    print(
        "\t--video=\"<file>\"\tread frames from a video file instead of the camera\n"
        "\t\t\t\t(useful for testing / benchmarking without a webcam)"
    )
    print(
        "\t--proc-size=<px>\tscale the long side to <px> before MediaPipe inference\n"
        "\t\t\t\t(0 disables scaling) [default: 0]"
    )
    print(
        "\t--log-level=<lvl>\tinfo | debug | warning [default: info]\n"
        "\t\t\t\tper-frame logs are only emitted at debug"
    )
    print(
        "\t--engine=<mode>\tholistic | minimal | auto [default: holistic]\n"
        "\t\t\t\tholistic = 原版全家桶；minimal = 只加载配置需要的模型。\n"
        "\t\t\t\t实测 holistic 更快更准，除非你在别的机器上复测过，否则别改"
    )

if __name__ == "__main__":
    fps = 8
    # 默认从 2(heavy) 降到 1(medium)：实测(M2 Max, mediapipe 0.10.14, 1280x720)
    # complexity 2→1 推理 76.0ms→32.6ms（快 2.3x），左右手检出 95→93 /120，
    # 精度几乎无损。要最高精度仍可显式 --complexity=2。
    complexity = 1
    dataDir = "./data"
    cvShow = False
    video = None
    procSize = 0
    logLevel = "info"
    engine = "holistic"
    opts, _ = getopt.getopt(sys.argv[1:], 'h', [
        'fps=', 'complexity=', 'help', 'data=', 'cvshow',
        'video=', 'proc-size=', 'log-level=', 'engine='
    ])
    for name, value in opts:
        if name in ['-h', '--help']:
            showHelp()
            exit(0)
        elif name == '--fps':
            try:
                fps = int(value)
            except:
                print(f"invalid fps {value}")
                exit(0)
            if fps < 0:
                print(f"invalid fps {value}")
                exit(0)
        elif name == '--complexity':
            try:
                complexity = int(value)
            except:
                print(f"invalid complexity {value}")
                exit(0)
            if not complexity in [0, 1, 2]:
                print(f"invalid complexity {value}")
                exit(0)
        elif name == '--data':
            dataDir = value
        elif name=='--cvshow':
            cvShow=True
        elif name == '--video':
            video = value
        elif name == '--proc-size':
            try:
                procSize = int(value)
            except:
                print(f"invalid proc-size {value}")
                exit(0)
            if procSize < 0:
                print(f"invalid proc-size {value}")
                exit(0)
        elif name == '--log-level':
            logLevel = (value or "").strip().lower()
            if not logLevel in ('info', 'debug', 'warning', 'error'):
                print(f"invalid log-level {value}")
                exit(0)
        elif name == '--engine':
            engine = (value or "").strip().lower()
            if not engine in ('holistic', 'minimal', 'auto'):
                print(f"invalid engine {value}")
                exit(0)

    # --video 的路径必须在 os.chdir(dataDir) 之前解析成绝对路径
    if video:
        video = os.path.abspath(video)
        if not os.path.exists(video):
            print(f"video not found: {video}")
            exit(0)

    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    configPath = os.path.join(dataDir, "config.json")
    if not os.path.exists(os.path.join(dataDir, "error.log")):
        with open(os.path.join(dataDir, "error.log"), "w") as f:
            f.write("")

    logging.basicConfig(
        level=getattr(logging, logLevel.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(dataDir, "error.log"), encoding='utf-8')
        ]
    )
    import mediapipe.python.solutions as sol

    logging.info(f"working in folder {dataDir}")
    if not os.path.exists(configPath):
        with open(configPath, "w") as f:
            f.write("[]")
        with sol.holistic.Holistic(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_complexity=0) as holistic:
            pass
        with sol.holistic.Holistic(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_complexity=1) as holistic:
            pass
        with sol.holistic.Holistic(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_complexity=2) as holistic:
            pass

        logging.info(f"Now you can modify {configPath} to custom your own task list")
        exit(0)

    from ValidateConfig import ValidateConfig

    os.chdir(dataDir)
    if not ValidateConfig("config.json"):
        exit(0)

    from TaskController import TaskController

    logging.info(f"use fps limit {fps}, model complexity {complexity}")
    controller = TaskController()
    used = controller.readConfig("config.json")
    logging.info(f"[mps] 配置实际需要的部位: {sorted(used)}"
                 f"（未列出的模型不会加载）")
    controller.startListen(fps, complexity, cvShow,
                           video=video, procSize=procSize, engine=engine)
