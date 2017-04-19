from PyQt4 import QtGui,QtCore
import sys
import ui_main
import numpy as np
import pyqtgraph
import SWHear

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from feature import *
from predict import *
import tensorflow as tf
import torch
import librosa


class ExampleApp(QtGui.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.grPCM.plotItem.showGrid(True, True, 0.7)
        self.audioAUG, self.updateCount = [], 0
        self.maxPCM=0
        self.ear = SWHear.SWHear(rate=44100,updatesPerSecond=20)
        self.ear.stream_start()
        self.predicted = ['initial']

    def accumlate(self, datastream):
        if self.updateCount ==  20:
            length = len(self.audioAUG)
            feature = sound_net_features(self.audioAUG)
            pred_conf = run_liblinear(feature)
            self.predicted.pop()
            self.predicted.append(pred_conf)
            self.audioAUG, self.updateCount = [], 0
        else:
            if self.updateCount == 0 :
                self.audioAUG = datastream
            else :
                self.audioAUG = np.concatenate((self.audioAUG, datastream), axis = 0)
            self.updateCount += 1
        return self.predicted[0]


    def update(self):
        if not self.ear.data is None and not self.ear.fft is None:
            pcmMax=np.max(np.abs(self.ear.data))
            if pcmMax>self.maxPCM:
                self.maxPCM=pcmMax
                self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
            self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
            pen=pyqtgraph.mkPen(color='b', width=2)
            className = self.accumlate(self.ear.data)
            self.grPCM.plot(self.ear.datax,self.ear.data,pen=pen,clear = True)#, name = className)
            font = QtGui.QGraphicsTextItem().font()
            font.setPointSizeF(22)
            tb = pyqtgraph.TextItem(className, color=(0, 0, 0), anchor=(-0.5, 0.5), border=None, fill=(255, 255, 255))
            tb.setFont(font)
            self.grPCM.addItem(tb)
        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()
    print("DONE")
