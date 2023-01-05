from AnalysisProcess.BasicAnalysisProcess import BasicAnalysisProcess
from AnalysisProcess.OperatorMethod.spatialFilter import TRCA
from AnalysisProcess.OperatorMethod.utils import ensembleData

import numpy as np
from psychopy import core
import pickle
import os

class TestingProcess(BasicAnalysisProcess):
    def __init__(self):
        self.algorithm = None
        self.state = None
        self.cacheData = None
        self.deposit = None #数据
        self.record = None #标签

        super().__init__()


    def loadModel(self):

        modelname = os.path.join(self.savepath,'models/model.pickle')
        with open(modelname,"rb") as fp:
            self.algorithm = pickle.load(fp)
        pass

    def run(self):

        startTime = core.getTime()
        # 同步系统,包含event
        while True:
            self.cacheData = self.streaming.readFixedData(self.startPoint, self.winLEN)
            if self.cacheData is not None:
                break

        # 读取数据
        epoch = self.cacheData

        import matplotlib.pyplot as plt
        # from scipy.fft import fft, fftfreq
        # y = epoch[0]
        # N = epoch.shape[-1]
        # yf = fft(y)
        # xf = fftfreq(N, 1/250)[:N//2]
        # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

        # 计算结果
        result = self.getResult(epoch)
        # 汇报结果
        self.controller.report(result)

        endTime = core.getTime()

        # 清空
        self.clear()
        # 模型评价
        self.controller.actualWin = endTime-startTime
        print('Time spend %f s' % self.controller.actualWin)

        self.controller.current_process = self.controller.wait_analysis_process

        return



