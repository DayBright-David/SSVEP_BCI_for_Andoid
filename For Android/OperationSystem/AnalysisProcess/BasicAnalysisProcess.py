from io import SEEK_CUR
from AnalysisProcess.OperatorMethod.spatialFilter import TRCA,fbCCA
from AnalysisProcess.OperatorMethod.utils import ITR, snsplot, changeDataFormat
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

class BasicAnalysisProcess:
    def __init__(self) -> None:
        pass


    @abstractmethod
    def initial(self, controller=None, config=None, streaming=None, messenger=None):

        # 3 essentials: controller, config, and data streaming client
        self.controller = controller
        self.config = config
        self.streaming = streaming
        self.messenger = messenger

        # 初始化算法：fbcca
        self.winLEN = config.winLEN
        self.srate = config.srate
        self.personName = config.personName
        self.prepareFolder()

        self.algorithm = fbCCA(winLEN=self.winLEN,lag=0)
        self.algorithm.fit()


        self.displayChar = config.displayChar
        self.startPoint = 0

    def prepareFolder(self):
        fatherAdd = 'OperationSystem/ResultStored'
        sonAdd = os.path.join(fatherAdd,self.personName)
        if not os.path.exists(sonAdd):
            os.makedirs(os.path.join(sonAdd,'images'))
            os.makedirs(os.path.join(sonAdd,'models'))
            os.makedirs(os.path.join(sonAdd,'data'))
            os.makedirs(os.path.join(sonAdd,'timelog'))
            os.makedirs(os.path.join(sonAdd,'selector'))
            os.makedirs(os.path.join(sonAdd,'src'))
            os.makedirs(os.path.join(sonAdd,'test'))
        self.savepath = sonAdd
        return


    @abstractmethod
    def run(self):

        pass


    def evalModels(self):
        # 适用于cue spelling,知道标准答案
        if (self.controller.currentEpochINX) % self.recordGap != 0:
            return
        else:
            # 确定当前block和epoch索引

            blockINX = self.controller.currentBlockINX
            self.controller.currentBlockINX += 1
            # 当前block的标准答案
            standardCue = self.controller.cues[blockINX]

            # 当前block的数据和标签
            UD = self.controller.UD
            UDData = UD['X']
            UDLabel = UD['y']

            UDData = changeDataFormat(UDData, window=self.winLEN)

            thisBlockTestData = UDData[-self.recordGap:,:,:]

            # 三种方法的结果汇报
            proposedAnswer = np.array(
                self.controller.resultsUntilNow)
            proposedAnswer = proposedAnswer[-self.recordGap:]

            self.controller.fbCCAModel.fit(thisBlockTestData)

            fbCCAAnswer = self.controller.fbCCAModel.predict(
                thisBlockTestData)

            weakAnswer = self.controller.weakSupModel.predict(
                thisBlockTestData)

            # weak model 需要加入所有数据重新训练
            self.controller.weakSupModel.fit(UDData, UDLabel)

            answers = [proposedAnswer, fbCCAAnswer, weakAnswer]

            # 映射成命令索引
            answers = [self.commandID[answer] for answer in answers]

            scoreEnsemble = [np.sum(
                standardCue == answer)/len(standardCue) for answer in answers]

            self._record(scoreEnsemble)

    def evalOnTest(self):
            # 适用于cue spelling,知道标准答案
        if (self.controller.currentEpochINX) % self.recordGap != 0:
            return
        else:
            # 确定当前block和epoch索引

            self.controller.currentBlockINX += 1

            # 当前block的数据和标签
            UDTest = self.controller.test
            UDTest_X = UDTest['X']
            UDTest_y = UDTest['y']
            UDTest_X, UDTest_y = changeDataFormat(
                UDTest_X, UDTest_y, window=self.winLEN)

            UDTrain = self.controller.UD
            UDTrain_X = UDTrain['X']
            UDTrain_y = UDTrain['y']
            UDTrain_X, UDTrain_y = changeDataFormat(
                UDTrain_X, UDTrain_y, window=self.winLEN)

            # 三种方法的结果汇报
            proposedAnswer = self.algorithm.predict(UDTest_X)

            self.controller.fbCCAModel.fit(UDTest_X)
            fbCCAAnswer = self.controller.fbCCAModel.predict(UDTest_X)

            # weak model 需要加入所有数据重新训练
            self.controller.weakSupModel.fit(UDTrain_X, UDTrain_y)
            weakAnswer = self.controller.weakSupModel.predict(
                UDTest_X)

            # 映射成命令索引

            answers = [proposedAnswer, fbCCAAnswer, weakAnswer]
            # 映射成命令索引
            answers = [self.commandID[answer] for answer in answers]

            scoreEnsemble = [np.sum(
                UDTest_y == answer)/len(UDTest_y) for answer in answers]

            self._record(scoreEnsemble)


    @abstractmethod
    def _record(self,scores):

        time = float(self.winLEN / self.srate)

        actualtime = float(self.controller.actualWin / self.srate)

        epochINX = self.controller.currentEpochINX
        subject = [self.personName for _ in range(3)]
        method = ['The Proposed', 'fbCCA', 'Weak Supervision TRCA']
        itr = [ITR(40, score, time) for score in scores]

        frame = pd.DataFrame({
            'score': scores,
            'ITR': itr,
            'scheme': method,
            'subject': subject,
            'window length': ['{win} s'.format(win=time) for _ in range(3)],
            'Epoch': ['{epoch}'.format(epoch=epochINX) for _ in range(3)]
        })

        # 保存结果
        self.controller.frames.append(frame)
        df = pd.concat(self.controller.frames, axis=0)
        df.to_csv(os.path.join(self.savepath, 'src/frames.csv'))

        # 画图
        snsplot(df, os.path.join(self.savepath, 'images/onlinePerformance.png'))

        # 保存数据
        with open(os.path.join(self.savepath, 'data/UDData'), "wb+") as fp:
            pickle.dump(self.controller.UD, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # 保存模型
        models = dict(
            Proposed=self.algorithm,
            fbCCA=self.controller.fbCCAModel,
            weakSupervision=self.controller.weakSupModel
        )
        with open(os.path.join(self.savepath, 'models/models.pickle'), "wb+") as fp:
            pickle.dump(models, fp, protocol=pickle.HIGHEST_PROTOCOL)


        return


    @abstractmethod
    def clear(self):
        self.cacheData = None
        return

    @abstractmethod
    def storeEpoch(self,y):

        cache = np.expand_dims(self.cacheData, axis=0)
        self.data['X'].append(cache)

        self.data['y'].append(int(y))
        self.data['coff'].append(self.algorithm.confidence)

        self.controller.resultsUntilNow.append(y)
        self.controller.currentEpochINX += 1
        return

    @abstractmethod
    def isEnough(self,data):
        dataLen = np.shape(data)[-1]
        return False if dataLen<self.winLEN else True

    @abstractmethod
    def getResult(self,data):
        result = self.algorithm.predict(data)
        return result[0]

