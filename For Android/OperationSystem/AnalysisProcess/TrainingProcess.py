from AnalysisProcess.BasicAnalysisProcess import BasicAnalysisProcess
from AnalysisProcess.OperatorMethod.spatialFilter import TRCA
from AnalysisProcess.OperatorMethod.utils import ensembleData

import numpy as np
import matplotlib.pyplot as plt
from psychopy import core
import pickle
import os

class TrainingProcess(BasicAnalysisProcess):
    def __init__(self):
        self.algorithm = None
        self.state = None
        self.selector = None
        self.cacheData = None
        self.deposit = None #数据
        self.record = None #标签

        super().__init__()


    def run(self):

        startTime = core.getTime()
        # 同步系统,包含event
        while True:
            self.cacheData = self.streaming.readFixedData(self.startPoint, self.winLEN)
            if self.cacheData is not None:
                break

        # 读取数据
        epoch = self.cacheData
        # 计算结果
        result = self.getResult(epoch)
        # 汇报结果s
        self.schema_controller.report(result)

        endTime = core.getTime()

        # 储存当前Epoch
        self._collectTraing()

        # 训练模型
        self._trainModel()

        # 清空
        self.clear()
        # 模型评价
        self.controller.actualWin = endTime-startTime
        print('Time spend %f s' % self.controller.actualWin)

        self.controller.current_process = self.controller.wait_analysis_process

        return

    def _collectTraing(self):

        blockINX = self.controller.currentBlockINX
        epochINX = self.controller.currentEpochINX
        # 标准答案
        y = self.controller.cues[blockINX].pop(0)

        cache = np.expand_dims(self.cacheData, axis=0)
        self.controller.traingUD['X'].append(cache)

        cropUD = []
        for cache in self.controller.traingUD['X']:
            cropUD.append(cache)

        self.controller.traingUD['X'] = cropUD
        self.controller.traingUD['y'].append(y)
        self.controller.traingUD['coff'].append(
            self.algorithm.confidence)

        if (epochINX+1) % self.recordGap == 0:

            self.controller.currentBlockINX += 1

            with open(os.path.join(self.savepath, 'test/testData.pickle'), "wb+") as fp:
                pickle.dump(self.controller.traingUD, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)

        self.controller.currentEpochINX += 1

        if (epochINX+1) % 240 == 0:
            self.controller.trainFlag = True

        return

    def _trainModel(self):

        if self.controller.trainFlag is not True:
            return

        else:
            model = TRCA(winLEN=self.winLEN)

            trainX = self.controller.traingUD['X']
            trainy = self.controller.traingUD['y']

            trainX = np.concatenate(trainX)
            trainy = np.array(trainy)

            model.fit(trainX, trainy)

            self.controller.algorithm = model

            with open(os.path.join(self.savepath, 'models/model.pickle'), "wb+") as fp:
                pickle.dump(model, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)

            self.schema_controller.trainEnd()

        return

