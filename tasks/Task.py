import os
import logging


class Task:

    @classmethod
    def validate(cls, task: dict, ids: list, sameIds: list):
        errorCount = 0
        warningCount = 0
        if not 'id' in task.keys():
            logging.error("Key Error: missing key 'id'")
            errorCount += 1
        elif type(task['id']) != str:
            logging.error(
                f"Type Error: 'id' expects a string, but found a {type(task['id'])}({task['id']}) instead"
            )
            errorCount += 1
        elif task['id'] in sameIds:
            logging.error(f"Value Error: duplicate 'id' detected ({task['id']})")

        if not 'nextTasks' in task.keys():
            logging.warning("Warning: missing key 'nextTasks', use [] as default")
            warningCount += 1
        elif type(task['nextTasks']) != list:
            logging.error(
                f"Type Error: 'nextTasks' expects a list, but found a {type(task['nextTasks'])}({task['nextTasks']}) instead"
            )
            errorCount += 1
        else:
            for i, subTask in enumerate(task['nextTasks']):
                if not 'operate' in subTask.keys():
                    logging.error(f"nextTasks[{i}] Key Error: missing key 'operate'")
                    errorCount += 1
                elif not subTask['operate'] in ['start', 'stop']:
                    logging.error(
                        f"nextTasks[{i}] Value Error: 'operate' key expects a value of either 'start' or 'stop'"
                    )
                    errorCount += 1
                if not 'id' in subTask.keys():
                    logging.error(f"nextTasks[{i}] Key Error: missing key 'id'")
                    errorCount += 1
                elif type(subTask['id']) != str:
                    logging.error(
                        f"nextTasks[{i}] Type Error: 'id' expects a string, but found a {type(subTask['id'])}({subTask['id']}) instead"
                    )
                    errorCount += 1
                elif not subTask['id'] in ids:
                    logging.error(
                        f"nextTasks[{i}] Value Error: unknown 'id' {subTask['id']}"
                    )

        if not 'start' in task.keys():
            logging.warning(f"warning: missing key 'start', use False as default")
            warningCount += 1
        elif not task['start'] in [True, False]:
            logging.error(
                f"Value Error: 'start' key expects a value of either True or False"
            )
            errorCount += 1

        return errorCount, warningCount

    def __init__(self,
                 controller: object,
                 id: str,
                 taskType: str,
                 nextTasks: list = [],
                 start: bool = True):
        self.controller = controller
        self.id = id
        self.taskType = taskType
        self.nextTasks = nextTasks
        self.start = start

    def activate(self, x):
        pass

    def deactivate(self, x):
        pass

    def process(self, x):
        logging.info(f"processing {self.id}")
        self.controller.deactivateTask(self.id, x)
        for i in self.nextTasks:
            if i['operate'] == 'start':
                self.controller.activateTask(i['id'], x)
            if i['operate'] == 'stop':
                self.controller.deactivateTask(i['id'], x)

    def _listen(self, x):
        raise NotImplementedError

    def listen(self, x):
        # 这里是**每帧每个激活任务**都会走的路径。原来写的是
        # `logging.info(..., end="")` —— 标准库 logging 根本不接受 `end`
        # 参数，所以只要有任何一个任务处于激活状态，跑到第
        # FPS_COUNT_FRAME*3 帧就会抛 TypeError 直接崩掉主程序；
        # 就算不崩，每帧写一条 INFO（stdout + error.log）也是纯浪费。
        # 改成 debug：默认不输出，需要排查时用 --log-level debug 打开。
        logging.debug(f"listening {self.id} type {self.taskType}")
        self._listen(x)
