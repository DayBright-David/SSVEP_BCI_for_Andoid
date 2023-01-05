import numpy as np
import os
from numpy.lib.function_base import append, diff, select
from numpy.lib.utils import source
from numpy.linalg.linalg import qr
import scipy.linalg as la
from scipy.optimize.zeros import results_c
from scipy.signal.filter_design import freqs
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import _class_cov, _cov, _class_means
from scipy import linalg
from sklearn.metrics import accuracy_score
from scipy import stats, signal
import math
from sklearn.linear_model import LinearRegression
from collections import Counter
import copy
import pandas as pd
# from pingouin import ttest
from statsmodels.stats.weightstats import ttest_ind
import seaborn as sns

class TRCA():
    def __init__(self, n_components=1, n_band=5, montage=40, winLEN=2, lag=35):

        self.n_components = n_components
        self.n_band = n_band
        self.montage = np.linspace(0, montage-1, montage).astype('int64')
        self.frequncy = np.linspace(8, 15.8, num=montage)
        self.phase = np.tile(np.arange(0, 2, 0.5)*math.pi, 10)
        self.srate = 250
        self.winLEN = round(self.srate*winLEN)
        self.lag = lag


    def fit(self, X, y):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """

        X = X[:,:,self.lag:self.lag+self.winLEN]

        self._classes = np.unique(y)

        self.filter = []
        self.evokeds = []
        for fbINX in range(self.n_band):
            # 每种信号都有不同的模版

            filter = []
            evokeds = []

            for classINX in self._classes:

                this_class_data = X[y == classINX]
                this_band_data = self.filterbank(
                    this_class_data, 250, freqInx=fbINX)
                evoked = np.mean(this_band_data, axis=0)
                evokeds.append(evoked)
                weight = self.computer_trca_weight(this_band_data)
                filter.append(weight[:, :self.n_components])

            self.filter.append(np.concatenate(filter, axis=-1))
            self.evokeds.append(np.stack(evokeds))

        self.filter = np.stack(self.filter)
        self.evokeds = np.stack(self.evokeds).transpose((1, 0, 2, 3))

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            shape is (n_epochs, n_sources, n_times).
        """

        enhanced = []
        for classINX in range(len(self._classes)):
            X_filtered = np.dot(self.filter[classINX][:, :self.n_components].T, X)
            X_filtered = X_filtered.transpose((1, 0, 2))
            X_filtered = np.stack(X_filtered[i].ravel() for i in range(X.shape[0]))
            enhanced.append(X_filtered)
        enhanced = np.concatenate(enhanced, axis=-1)

        return enhanced

    def fit_transform(self,X,y):

        return self.fit(X,y).transform(X)

    def IRF(self,X,y):

        X = X[:,:,:self.winLEN]
        N = X.shape[-1]
        # extract template
        labels = np.unique(y)
        conNUM = len(labels)
        ave = np.zeros((conNUM,N))
        for i,l in enumerate(labels):
            ave[i] = X[y==l].mean(axis=(0,1))
        
        # 生成和数据等长的正弦刺激
        sti = self.sine(N)[labels]

        score = np.zeros((conNUM,N))
        for i,(t1,t2) in enumerate(zip(sti,ave)):
            s = np.correlate(t2-t2.mean(),t1-t1.mean(),mode='full')
            s = s / (N * t2.std() * t1.std())
            score[i] = s[N-1:]

        ave_score = score.mean(axis=0)
        self.lag = np.argmax(ave_score)

        return score,self.lag


    def sine(self,winLEN):
        sine = []
        win = winLEN
        t = np.arange(0, (win/self.srate), 1/self.srate)
        for f,p in zip(self.frequncy,self.phase):
            sine.append([np.sin(2*np.pi*i*f+p) for i in t])
        self.sti = np.stack(sine)
        return self.sti

    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        # crop test data according to predefined latency

        X = X[:,:,self.lag:self.lag+self.winLEN]

        result = []
        cropLen = X.shape[-1]  # 允许测试数据和模版不一样 但是测试数据必须比是模版短
        H = np.zeros(X.shape[0])
        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        for epochINX, epoch in enumerate(X):

            r = np.zeros((self.n_band, len(self.montage)))

            for fbINX in range(self.n_band):

                epoch_band = np.squeeze(self.filterbank(epoch, 250, fbINX))

                for (classINX, evoked) in zip(self._classes, self.evokeds):

                    template = np.squeeze(evoked[fbINX, :, :cropLen])
                    w = np.squeeze(self.filter[fbINX, :])
                    rtemp = np.corrcoef(
                        np.dot(epoch_band.T, w).reshape(-1), np.dot(template.T, w).reshape(-1))
                    r[fbINX, classINX] = rtemp[0, 1]

            rho = np.dot(fb_coefs, r)

            missing = np.setdiff1d(
                self.montage, self._classes)  # missing filter
            rho[:, missing] = None

            # hypothesis testing

            target = np.nanargmax(rho)
            rhoNoise = np.delete(rho, target)
            rhoNoise = np.delete(rhoNoise, np.isnan(rhoNoise))
            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])
            result.append(target)

        self.confidence = H

        return np.stack(result)

    def residual(self, x, result):

        evoked = self.evokeds[int(result)]

        for fbINX in range(self.n_band):

            epoch_band = np.squeeze(self.filterbank(x, 250, fbINX))
            template = np.squeeze(evoked[fbINX, :, :])
            w = np.squeeze(self.filter[fbINX, :])
            t = np.dot(template.T, w).reshape(-1)
            s = np.dot(epoch_band.T, w).reshape(-1)
            residual = t-s
            stats.ttest_1samp(residual, 0.0)

        pass

    def score(self, X, y):

        return accuracy_score(y, self.predict(X))

    def dyStopping(self, X, former_win):

        # 判断要设置什么窗
        srate = self.srate
        p_val = 0.002

        dyStopping = np.arange(0.4, former_win+0.1, step=0.2)

        for ds in dyStopping:

            ds = int(ds*srate)
            self.predict(X[:, :, :ds+self.lag])

            score = self.confidence < p_val
            pesudo_acc = np.sum(score != 0)/len(score)
            print('mean confidence:', self.confidence.mean())
            print('pesudo_acc {pesudo_acc} %'.format(pesudo_acc=pesudo_acc))

            # if pesudo_acc >= 0.85:
            #     boostWin = ds
            #     break
        
            if self.confidence.mean() < p_val:
                boostWin = ds
                break

        # 难分的epoch下一次继续
        n = np.argsort(self.confidence)
        difficult = X[n[-5:]]

        if not 'boostWin' in locals().keys():
            boostWin = int(dyStopping[-1]*srate)

        return (boostWin/srate), difficult

    def recordCoff(self, X, y, subINX, adINX, frames):
        # 判断要设置什么窗

        srate = 250
        former_win = 750
        dyStopping = np.arange(0.6, (former_win/srate)+0.1, step=0.2)

        for ds in dyStopping:
            ds = int(ds*srate)
            labels = self.predict(X[:, :, :ds])
            frame = pd.DataFrame({
                'coff': self.confidence,
                'tw': [str(ds) for _ in range(len(X))],
                'personID': ['S{INX}'.format(INX=subINX) for _ in range(len(X))],
                'predicted label': labels,
                'true label': y,
                'correct': labels == y,
                'adptation INX': [adINX for _ in range(len(X))]

            })
            frames.append(frame)

        df = pd.concat(frames, axis=0)
        df.to_csv('dynamics/S{inx}.csv'.format(inx=subINX))

        return frames

    def filterbank(self, x, srate, freqInx):

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

        srate = srate/2

        Wp = [passband[freqInx]/srate, 90/srate]
        Ws = [stopband[freqInx]/srate, 100/srate]
        [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x)) == 2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX, :] = signal.filtfilt(
                    B, A, x[channelINX, :])
            filtered_signal = np.expand_dims(filtered_signal, axis=-1)
        else:
            for epochINX, epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX, channelINX, :] = signal.filtfilt(
                        B, A, epoch[channelINX, :])

        return filtered_signal

    def computer_trca_weight(self, eeg):
        """
        Input:
            eeg : Input eeg data (# of targets, # of channels, Data length [sample])
        Output:
            W : Weight coefficients for electrodes which can be used as a spatial filter.
        """
        epochNUM, self.channelNUM, _ = eeg.shape

        S = np.zeros((self.channelNUM, self.channelNUM))

        for epoch_i in range(epochNUM):

            x1 = np.squeeze(eeg[epoch_i, :, :])
            x1 = x1 - np.mean(x1, axis=1, keepdims=True)

            for epoch_j in range(epoch_i+1, epochNUM):
                x2 = np.squeeze(eeg[epoch_j, :, :])
                x2 = x2 - np.mean(x2, axis=1, keepdims=True)
                S = S + np.dot(x1, x2.T) + np.dot(x2, x1.T)

        UX = np.stack([eeg[:, i, :].ravel() for i in range(self.channelNUM)])
        UX = UX - np.mean(UX, axis=1, keepdims=True)
        Q = np.dot(UX, UX.T)

        C = np.linalg.inv(Q).dot(S)
        _, W = np.linalg.eig(C)

        return W


class fbCCA():
    def __init__(self, n_components=1, n_band=5, srate=250, conditionNUM=40,lag=35,winLEN=1):
        self.n_components = n_components
        self.n_band = n_band
        self.srate = srate
        self.conditionNUM = conditionNUM
        self.frequncy_info = np.linspace(8, 15.8, num=self.conditionNUM)
        self.lag = lag
        self.winLEN = int(self.srate*winLEN)
        self._classes = np.linspace(
            0, self.conditionNUM-1, num=self.conditionNUM).astype('int')

    def fit(self, X=[], y=[]):

        epochLEN = self.winLEN
        
        sineRef = self.get_reference(
            self.srate, self.frequncy_info, n_harmonics=5, data_len=epochLEN)

        self.evokeds = sineRef

        return self

    def predict(self, X):
        
        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)
            
        X = X[:,:,self.lag:self.lag+self.winLEN]
        result = []

        H = np.zeros(X.shape[0])
        corr = np.zeros(X.shape[0])

        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        for epochINX, epoch in enumerate(X):
            r = np.zeros((self.n_band, self.conditionNUM))
            cca = CCA(n_components=1)
            for fbINX in range(self.n_band):
                epoch_band = np.squeeze(self.filterbank(epoch, 250, fbINX))
                for (classINX, evoked) in zip(self._classes, self.evokeds):
                    u, v = cca.fit_transform(evoked.T, epoch_band.T)
                    rtemp = np.corrcoef(u.T, v.T)
                    r[fbINX, classINX] = rtemp[0, 1]
            rho = np.dot(fb_coefs, r)

            # snr = np.power(rho,2)/(1-np.power(rho,2)) * featureNUM
            target = np.nanargmax(rho)
            # snrNoise = np.delete(snr,target)
            rhoNoise = np.delete(rho, target)

            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])
            corr[epochINX] = rho[0, target]

            result.append(target)

        self.confidence = H
        self.corr = corr

        return np.stack(result)

    def filterbank(self, x, srate, freqInx):

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

        srate = srate/2
        Wp = [passband[freqInx]/srate, 90/srate]
        Ws = [stopband[freqInx]/srate, 100/srate]
        [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x)) == 2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX, :] = signal.filtfilt(
                    B, A, x[channelINX, :])
            filtered_signal = np.expand_dims(filtered_signal, axis=-1)
        else:
            for epochINX, epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX, channelINX, :] = signal.filtfilt(
                        B, A, epoch[channelINX, :])

        return filtered_signal

    def get_reference(self, srate, frequncy_info, n_harmonics, data_len):

        t = np.arange(0, (data_len/srate), 1/srate)
        reference = []

        for j in range(n_harmonics):
            harmonic = [np.array([np.sin(2*np.pi*i*frequncy_info*(j+1)) for i in t]),
                        np.array([np.cos(2*np.pi*i*frequncy_info*(j+1)) for i in t])]
            reference.append(harmonic)
        reference = np.stack(reference[i] for i in range(len(reference)))
        reference = np.reshape(
            reference, (2*n_harmonics, data_len, len(frequncy_info)))
        reference = np.transpose(reference, (-1, 0, 1))

        return reference

    def score(self, X, y):

        return accuracy_score(y, self.predict(X))


class LstTRCA(TRCA):

    def adaptation(self, source, appendixs):

        evoked = source.mean(axis=0)
        appendix = []
        lst = LinearRegression()
        for target in appendixs:

            lst.fit(target.T, evoked.T)
            adapted_ = lst.predict(target.T)
            appendix.append(adapted_.T)

        appendix = np.stack(appendix)
        full = np.concatenate((source, appendix), axis=0)
        return full


    def fit(self, S, y1, X, y2):

        S = S[:, :, self.lag:self.lag+self.winLEN]
        X = X[:, :, self.lag:self.lag+self.winLEN]

        fragment = np.unique(y1)
        self._classes = np.unique(y2)
        # 只取出符合条件的个人数据
        exist = [key for key, value in Counter(y1).items() if value >= 2]

        self.filter = []
        self.evokeds = []

        for fbINX in range(self.n_band):

            # 每种信号都有不同的模版
            filter = []
            evokeds = []

            for classINX in self._classes:
                # enough personal data:adaption
                if np.any(classINX == exist):

                    source, appendix = S[y1 == classINX], X[y2 == classINX]
                    this_class_source = self.filterbank(
                        source, self.srate, freqInx=fbINX)
                    this_class_append = self.filterbank(
                        appendix, self.srate, freqInx=fbINX)
                    this_class_data = self.adaptation(
                        this_class_source, this_class_append)

                # not enough personal data: fusion
                elif not np.any(classINX == exist) and np.any(classINX == fragment):
                    source, appendix = S[y1 == classINX], X[y2 == classINX]
                    this_class_data = np.concatenate(
                        (source, appendix), axis=0)
                    this_class_data = self.filterbank(
                        this_class_data, self.srate, freqInx=fbINX)
                # personal data doesn’t exist : fusion
                else:
                    this_class_data = X[y2 == classINX]
                    this_class_data = self.filterbank(
                        this_class_data, self.srate, freqInx=fbINX)

                evoked = np.mean(this_class_data, axis=0)
                evokeds.append(evoked)

                weight = self.computer_trca_weight(this_class_data)
                filter.append(weight[:, :self.n_components])

            self.filter.append(np.concatenate(filter, axis=-1))
            self.evokeds.append(np.stack(evokeds))

            # the missing template are set to nan
        self.filter = np.stack(self.filter)
        self.evokeds = np.stack(self.evokeds).transpose((1, 0, 2, 3))

        return self


class TDCA(TRCA, fbCCA):

    def __init__(self, n_components=8, n_band=5, montage=40):
        self.srate = 250
        self.conditionNUM = 40
        self.lags = np.linspace(0, 4, 5).astype(int)
        self.frequncy_info = np.linspace(8, 15.8, num=self.conditionNUM)
        super().__init__(n_components=n_components, n_band=n_band, montage=montage)

        
    def fit(self, X, y):
        # X: 160*9*750
        self._classes = np.unique(y)
        self.filters = []
        self.evokeds = []
        self.epochNUM, self.channelNUM, N = X.shape

        self.winLEN = N-len(self.lags)

        # 先子带滤波，否则会引入很大计算量
        X = self.augumentation(X)

        X = np.transpose(X, axes=(1, 0, -2, -1))

        # 求正弦模版投影的空域滤波器
        self.P = self.sineQrDecomposition(X)

        augumentX = []

        for fbX in X:

            augumentClass = []

            for classINX, sineP in zip(self._classes, self.P):
                this_class_data = fbX[y == classINX]

                augumentEpoch = []

                for epoch in this_class_data:

                    epoch = np.hstack((
                        epoch, np.dot(epoch, sineP.T)
                    ))

                    augumentEpoch.append(epoch)

                augumentClass.append(np.stack(augumentEpoch))

            augumentX.append(augumentClass)

        # conditionNUM*epochNUM*fbNUM*chnNUM*N
        # 如果每个condition的数目不一样，就不能stack
        # augumentX = np.stack(augumentX)

        self.computer_tdca_weight(augumentX)

        return self

    def predict(self, X):



        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        result = []
        cropLen = X.shape[-1]  # 允许测试数据和模版不一样 但是测试数据必须比是模版短

        self.winLEN = cropLen-len(self.lags)
        
        H = np.zeros(X.shape[0])
        fb_coefs = np.expand_dims(
            np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)

        X = self.augumentation(X)

        for epochINX, epoch in enumerate(X):

            r = np.zeros((self.n_band, len(self.montage)))

            for (classINX, evoked, sineP) in zip(self._classes, self.evokeds, self.P):

                for fbINX, (fbEvoked, fbEpoch, filter) in enumerate(zip(evoked, epoch, self.filters)):

                    fbEpoch = np.hstack((
                        fbEpoch, np.dot(fbEpoch, sineP[:self.winLEN,:self.winLEN].T)
                    ))

                    fbEvoked = np.array_split(fbEvoked,indices_or_sections=2,axis=-1)
                    fbEvoked = np.hstack((
                        fbEvoked[0][:, :self.winLEN],
                        fbEvoked[1][:, :self.winLEN],
                    ))
                    rtemp = np.corrcoef((
                        np.dot(fbEpoch.T, filter).reshape(-1),
                        np.dot(fbEvoked.T, filter).reshape(-1)
                    ))
                    r[fbINX, classINX] = rtemp[0, 1]

            rho = np.dot(fb_coefs, r)
            missing = np.setdiff1d(
                self.montage, self._classes)  # missing filter
            rho[:, missing] = None

            target = np.nanargmax(rho)
            rhoNoise = np.delete(rho, target)
            rhoNoise = np.delete(rhoNoise, np.isnan(rhoNoise))
            _, H[epochINX], _ = ttest_ind(rhoNoise, [rho[0, target]])
            result.append(target)

        self.confidence = H

        return np.stack(result)

    def computer_tdca_weight(self, augumentX):

        augumentEvoked = []
        for fbs in augumentX:
            augumentEvoked.append([con.mean(axis=0) for con in fbs])
        augumentEvoked = np.stack(augumentEvoked)

        for (fbEvoked, fbEpochs) in zip(augumentEvoked, augumentX):
            # norm
            fbEvoked = fbEvoked-np.mean(fbEvoked, axis=-1, keepdims=True)
            fbEvokedFeature = -np.mean(fbEvoked, axis=0, keepdims=True)
            betwClass = fbEvoked-fbEvokedFeature
            betwClass = np.concatenate(betwClass, axis=1)
            # norm
            fbEpochs = [
                this_class-np.mean(this_class, axis=-1, keepdims=True) for this_class in fbEpochs
            ]
            allClassEvoked = [
                this_class-np.mean(this_class, axis=0, keepdims=True) for this_class in fbEpochs
            ]

            allClassEvoked = [np.transpose(this_class, axes=(
                1, 2, 0)) for this_class in allClassEvoked]
            allClassEvoked = [np.reshape(
                this_class, (self.channelNUM, -1), order='F') for this_class in allClassEvoked]
            allClassEvoked = np.hstack(allClassEvoked)

            Hb = betwClass/math.sqrt(self.conditionNUM)
            Hw = allClassEvoked/math.sqrt(self.epochNUM)
            Sb = np.dot(Hb, Hb.T)
            Sw = np.dot(Hw, Hw.T)+0.001*np.eye(Hw.shape[0])

            # inv(Sw)*B
            C = np.linalg.inv(Sw).dot(Sb)
            _, W = np.linalg.eig(C)
            _, W = la.eig(C)

            # tmd又是反的？
            self.filters.append(W[:, :self.n_components])

        self.evokeds = np.transpose(augumentEvoked, axes=(1, 0, -2, -1))

        return

    def augumentation(self, X):

        augumentX = []
        for epoch in X:
            fbedEpoch = []
            for fbINX in range(self.n_band):
                fbEpoch = self.filterbank(epoch, self.srate, fbINX)
                lagedEpoch = self.lagEpoch(fbEpoch, self.lags)
                fbedEpoch.append(lagedEpoch)
            fbedEpoch = np.concatenate(fbedEpoch, axis=-1)
            augumentX.append(fbedEpoch)

        augumentX = np.stack(augumentX)
        augumentX = np.transpose(augumentX, axes=(0, -1, 1, 2))
        self.channelNUM = self.channelNUM*len(self.lags)
        return augumentX

    def lagEpoch(self, epoch, lags):
        # X : epochNUM*fbNUM*chnNUM*N
        winLEN = self.winLEN

        lagedX = []

        for lag in lags:
            # 这里需要确定下，是不是需要多拿一个点的数据？
            lagedX.append(epoch[:, lag:lag+winLEN, :])

        lagedX = np.concatenate(lagedX, axis=0)

        return lagedX

    def sineQrDecomposition(self, eeg):
        epochlen = np.shape(eeg)[-1]
        reference = self.get_reference(
            self.srate, self.frequncy_info, self.n_band, epochlen)

        P = []
        for ref in reference:
            [Q, _] = qr(ref.T)
            P.append(np.dot(Q, Q.T))
        return P


class LstTDCA(TDCA, LstTRCA):
    def __init__(self, n_components=1, n_band=5, montage=40):
        super().__init__(n_components=n_components, n_band=n_band, montage=montage)

    def joinData(self, S, y1, X, y2):

        X_ = []
        y_ = []

        S = S[:, :, self.lag:self.lag+self.winLEN]
        X = X[:, :, self.lag:self.lag+self.winLEN]
        fragment = np.unique(y1)
        self._classes = np.unique(y2)
        # 只取出符合条件的个人数据
        exist = [key for key, value in Counter(y1).items() if value >= 2]

        for classINX in self._classes:
            # enough personal data:adaption
            if np.any(classINX == exist):

                source, appendix = S[y1 == classINX], X[y2 == classINX]
                this_class_data = self.adaptation(
                    source, appendix)

            # not enough personal data: fusion
            elif not np.any(classINX == exist) and np.any(classINX == fragment):
                source, appendix = S[y1 == classINX], X[y2 == classINX]
                this_class_data = np.concatenate(
                    (source, appendix), axis=0)

            # personal data doesn’t exist : fusion
            else:
                this_class_data = X[y2 == classINX]

            X_.append(this_class_data)
            y_.append(np.tile(classINX, len(this_class_data)))

        return np.concatenate(X_), np.concatenate(y_)

    def fit(self, S, y1, X, y2):
        X, y = self.joinData(S, y1, X, y2)

        return super().fit(X, y)


class cusTRCA(TRCA):

    def __init__(self,winLEN=1,lag=35):
        # 扩增模版的倍数：0就是不扩增
        self.ampNUM = 0
        super().__init__(winLEN=winLEN,lag=lag)


    def fit(self, X, y):

        # decide latency
        self.IRF(X, y)

        self._classes = np.unique(y)

        # crop
        X = X[:, :, self.lag:self.lag+self.winLEN]

        epochN,chnN,_= X.shape

        self.ref = fbCCA().get_reference(self.srate, self.frequncy, self.n_band, self.winLEN)

        # filterbank
        X = self.subBand(X)
        
        self.filter = np.zeros((self.n_band,len(self._classes), chnN))
        self.evokeds = np.zeros((epochN,self.n_band ,chnN, self.winLEN))

        for classINX in self._classes:

            this_class_data = X[y == classINX]
            this_class_data = this_class_data.transpose((1,0,-2,-1))

            for fbINX, this_band in enumerate(this_class_data):
                # personal template
                self.evokeds[classINX, fbINX] = np.mean(this_band, axis=0)
                # trca weight
                weight = self.computer_trca_weight(this_band)
                # fbCCA part
                self.filter[fbINX, classINX] = weight[:, :self.n_components].squeeze()
        
        return self


    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        
        # filterbank
        X = self.subBand(X)

        # data augumentation
        Xs = self.cropData(X)

        epochN,_,_,N = Xs[0].shape
        

        fb_coefs = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)
        # coff for personal template 
        personR = np.zeros((epochN, len(Xs), len(self.montage)))
        # coff for sine/cosine template
        refR = np.zeros_like(personR)

        for driftINX,Xd in enumerate(Xs):
            # every drift
            for epochINX,epoch in enumerate(Xd):
                
                r1 = np.zeros((self.n_band, len(self.montage))) #trca
                r2 = copy.deepcopy(r1) #fbcca

                for fbINX, this_band in enumerate(epoch):

                        cca = CCA(n_components=1)

                        for (classINX, ref) in zip(self.montage,self.ref):

                            if classINX in self._classes:
                                # test epoch might be shorter than evokeds

                                evoked = self.evokeds[self._classes==classINX].squeeze()
                                template = evoked[fbINX, :, :N]
                                w = self.filter[fbINX,:].T
                                #trca : correlation w/ personal template
                                coffPerson = np.corrcoef(
                                    np.dot(this_band.T, w).reshape(-1), 
                                    np.dot(template.T, w).reshape(-1)
                                    )
                                r1[fbINX, classINX] = coffPerson[0, 1]

                            ref = ref[:,:N]
                            # fbcca : correlation w/ sine/cosine template
                            u, v = cca.fit_transform(ref.T, this_band.T)
                            coffRef = np.corrcoef(u.T, v.T)

                            r2[fbINX, classINX] = coffRef[0, 1]
                # fb coff dot product :[1,40]
                r1 = fb_coefs.dot(r1)
                r2 = fb_coefs.dot(r2)

                personR[epochINX, driftINX] = r1
                refR[epochINX,driftINX] = r2
        
        # missing template
        missing = np.setdiff1d(self.montage, self._classes)  # missing filter
        personR[:,:, missing] = 0

        # evaluate confidence

        addR = personR + refR
        H, predicts = self.evalConfidence(addR)
        # H1,p1: trca
        H1, predicts1 = self.evalConfidence(personR)
        # H2,p2: fbCCA
        H2, predicts2 = self.evalConfidence(refR)
        
        # choose the winner
        finalR = [predicts[i, np.argmin(H[i, :])] for i in range(H.shape[0])]

        self.confidence = dict(
            trca=H1,
            fbcca=H2,
            combined=H
        )
        
        self.results = dict(
            trca=predicts1,
            fbcca=predicts2,
            combined=predicts
        )

        return finalR

    def evalConfidence(self,coff):
        epochN,driftN,_ = coff.shape
        H = np.zeros((epochN,driftN))
        predicts = copy.deepcopy(H)
        # each epoch
        for epochINX,c in enumerate(coff):
            # each drift
            for driftINX,rho in enumerate(c):

                target = np.nanargmax(rho)
                predicts[epochINX,driftINX] = target

                rhoNoise = np.delete(rho, target)
                rhoNoise = np.delete(rhoNoise, np.isnan(rhoNoise))

                _, H[epochINX, driftINX], _ = ttest_ind(rhoNoise, [rho[target]])

        return H,predicts


    def cropData(self, X):
        ampNUM = self.ampNUM
        self.amp = np.arange(-ampNUM, ampNUM+1)
        ampX = []

        for drift in self.amp:
            ampX.append(X[:, :, :, self.lag+drift:self.lag+drift+self.winLEN])
        return ampX


    def subBand(self,X):
        
        Bands = []

        for epoch in X:

            filtered = []

            for fb in range(self.n_band):
            
                filtered.append(np.squeeze(self.filterbank(epoch,self.srate,fb)))

            Bands.append(np.stack(filtered))

        Bands = np.stack(Bands)

        return Bands


class tTRCA(cusTRCA,TRCA):
    # sin/cos templates with transfer templlates

    def __init__(self, winLEN=1,lag=35):
        super().__init__(winLEN=winLEN, lag=lag)


    def fit(self, X, y):

        self._classes = np.unique(y)

        # crop
        X = X[:, :, self.lag:self.lag+self.winLEN]

        epochN, chnN, _ = X.shape

        self.ref = fbCCA().get_reference(self.srate, self.frequncy, self.n_band, self.winLEN)

        # filterbank
        X = self.subBand(X)

        self.filter = np.zeros((self.n_band, len(self._classes), chnN))
        self.evokeds = np.zeros((len(self._classes), self.n_band, chnN, self.winLEN))

        for classINX in self._classes:

            this_class_data = X[y == classINX]
            this_class_data = this_class_data.transpose((1, 0, -2, -1))

            for fbINX, this_band in enumerate(this_class_data):
                # personal template
                self.evokeds[self._classes==classINX, fbINX] = np.mean(this_band, axis=0)
                # trca weight
                weight = self.computer_trca_weight(this_band)
                # fbCCA part
                self.filter[fbINX, self._classes==classINX] = weight[:,
                                                      :self.n_components].squeeze()

        return self


    def dyStopping(self, X, former_win):

        # 判断要设置什么窗
        p_val = 0.005

        dyStopping = np.arange(0.4, former_win+0.1, step=0.2)

        for ds in dyStopping:

            ds = int(ds*self.srate)
            self.predict(X[:, :, :ds+self.lag])

            score = self.confidence < p_val
            pesudo_acc = np.sum(score != 0)/len(score)
            print('mean confidence:', self.confidence.mean())
            print('pesudo_acc{pesudo_acc}s'.format(pesudo_acc=pesudo_acc))

            # if pesudo_acc >= 0.9:
            #     boostWin = float(ds/self.srate)
            #     break
        
            if self.confidence.mean() < p_val:
                boostWin = float(ds/self.srate)
                break

        # 难分的epoch下一次继续
        n = np.argsort(self.confidence)
        difficult = X[n[-5:]]

        if not 'boostWin' in locals().keys():
            boostWin = float('%.1f'%dyStopping[-1])

        return boostWin, difficult


    def predict(self, X):

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=0)

        
        # filterbank
        X = self.subBand(X)

        # data augumentation
        Xs = self.cropData(X)

        epochN,_,_,N = Xs[0].shape
        

        fb_coefs = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)
        # coff for personal template 
        personR = np.zeros((epochN, len(Xs), len(self.montage)))
        # coff for sine/cosine template
        refR = np.zeros_like(personR)

        for driftINX,Xd in enumerate(Xs):
            # every drift
            for epochINX,epoch in enumerate(Xd):
                
                r1 = np.zeros((self.n_band, len(self.montage))) #trca
                r2 = copy.deepcopy(r1) #fbcca

                for fbINX, this_band in enumerate(epoch):

                        cca = CCA(n_components=1)

                        for (classINX, ref) in zip(self.montage,self.ref):

                            if classINX in self._classes:
                                # test epoch might be shorter than evokeds

                                evoked = self.evokeds[self._classes==classINX].squeeze()
                                template = evoked[fbINX, :, :N]
                                w = self.filter[fbINX,:].T
                                #trca : correlation w/ personal template
                                coffPerson = np.corrcoef(
                                    np.dot(this_band.T, w).reshape(-1), 
                                    np.dot(template.T, w).reshape(-1)
                                    )
                                r1[fbINX, classINX] = coffPerson[0, 1]

                            ref = ref[:,:N]
                            # fbcca : correlation w/ sine/cosine template
                            u, v = cca.fit_transform(ref.T, this_band.T)
                            coffRef = np.corrcoef(u.T, v.T)

                            r2[fbINX, classINX] = coffRef[0, 1]
                # fb coff dot product :[1,40]
                r1 = fb_coefs.dot(r1)
                r2 = fb_coefs.dot(r2)

                personR[epochINX, driftINX] = r1
                refR[epochINX,driftINX] = r2
        
        # missing template
        missing = np.setdiff1d(self.montage, self._classes)  # missing filter
        personR[:,:, missing] = 0

        # evaluate confidence

        addR = personR + refR

        H, predicts = self.evalConfidence(addR)

        self.confidence = H
        # choose the winner
        finalR = [predicts[i, np.argmin(H[i, :])] for i in range(H.shape[0])]

        return finalR



class tTRCAwLST(tTRCA,LstTDCA):

    def __init__(self,winLEN=1,lag=35,n_components=1, n_band=5, montage=40):
        # 多继承

        self.n_components = n_components
        self.n_band = n_band
        self.montage = np.linspace(0, montage-1, montage).astype('int64')
        self.frequncy = np.linspace(8, 15.8, num=montage)
        self.phase = np.tile(np.arange(0, 2, 0.5)*math.pi, 10)
        self.srate = 250
        self.winLEN = round(self.srate*winLEN)
        self.lag = lag
        self.ampNUM = 0

    
    def fit(self, S, y1, X, y2):

        X, y = self.joinData(S, y1, X, y2)

        self._classes = np.unique(y)

        # crop
        epochN, chnN, _ = X.shape

        self.ref = fbCCA().get_reference(
            self.srate, self.frequncy, self.n_band, self.winLEN)

        # filterbank
        X = self.subBand(X)

        self.filter = np.zeros((self.n_band, len(self._classes), chnN))
        self.evokeds = np.zeros((len(self._classes), self.n_band, chnN, self.winLEN))

        for classINX in self._classes:

            this_class_data = X[y == classINX]
            this_class_data = this_class_data.transpose((1, 0, -2, -1))

            for fbINX, this_band in enumerate(this_class_data):
                # personal template
                self.evokeds[classINX, fbINX] = np.mean(this_band, axis=0)
                # trca weight
                weight = self.computer_trca_weight(this_band)
                # fbCCA part
                self.filter[fbINX, classINX] = weight[:,
                                                      :self.n_components].squeeze()

        return self

if __name__ == '__main__':

    trainX = np.random.random(size=(10,9,500))
    trainy = np.arange(10, trainX.shape[0]+10)


    testX = trainX
    testy = np.arange(10,testX.shape[0]+10)

    model = tTRCA()
    model.fit(trainX, trainy)
    model.predict(testX)
