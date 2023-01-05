from OperationSystem.AnalysisProcess.TestingProcess import TestingProcess
from OperationSystem.AnalysisProcess.WaitAnalysisProcess import WaitAnalysisProcess
from OperationSystem.AnalysisProcess.TrainingProcess import TrainingProcess
import pandas as pd
import pickle
import datetime
import os

class AnalysisController:
    def __init__(self):
        self.current_process = None
        self.algorithm = None

        # user dependent data and weak label

        self.frames = []
        self.timelog = []

        self.currentBlockINX=0
        self.currentEpochINX = 0

        self.resultsUntilNow = []
        self.actualWin = None

        self.trainFlag = True

    def initial(self, config, streaming,messenger):

        self.messenger = messenger
        # 个人数据
        self.data = dict(
            X = [],# data
            y = [], # label
        )

        self.cues = config.cue

        # 训练阶段
        self.training_process = TrainingProcess()
        self.training_process.initial(self, config, streaming, messenger)

        # 测试阶段
        self.testing_process = TestingProcess()
        self.testing_process.initial(self, config, streaming, messenger)

        # 等待下一次处理
        self.wait_analysis_process = WaitAnalysisProcess()
        self.wait_analysis_process.initial(self, config, streaming, messenger)

        self.current_process = self.wait_analysis_process

        return self

    def update(self):

        # 每次需要将模型更新至最新
        self.select_analysis_process.sync(self.algorithm)

        self.adaptive_analysis_process.sync(self.algorithm)

        predictCommand = self.current_process.commandID[[self.resultsUntilNow[-1]]]

        log = pd.DataFrame({
            'epoch': [self.currentEpochINX],
            'time': ['%s'%datetime.datetime.now()],
            'eureka': [self.selector.eurekaMoment],
            'predict': [self.resultsUntilNow[-1]],
            'predictLabel': predictCommand,
            # 'trueLabel': [self.cues[self.currentBlockINX][self.currentEpochINX-1]],
            'frequency': self.current_process.freValue[predictCommand],
            'win': [self.current_process.winLEN],
            'actualWin':[self.actualWin]
        })
        self.timelog.append(log)
        timelogFrames = pd.concat(self.timelog, axis=0)
        timelogFrames.to_csv(os.path.join(
            self.current_process.savepath, 'timelog/timelog.csv'))

    def report(self, resultID):
        message = 'RSLT:'+str(int(resultID))
        self.messenger.send_exchange_message(message)


    def run(self):
        # algorithm需要在各个状态之间传递
        self.current_process.run()




