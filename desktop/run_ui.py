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

class ExampleApp(QtGui.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.grFFT.plotItem.showGrid(True, True, 0.7)
        self.grPCM.plotItem.showGrid(True, True, 0.7)
        self.audioAUG, self.updateCount = [], 0
        self.maxFFT=0
        self.maxPCM=0
        self.ear = SWHear.SWHear(rate=44100,updatesPerSecond=20)
        self.ear.stream_start()
        self.predicted = ['initial']

    def accumlate(self, datastream):
        if self.updateCount == 20 :
            features = get_feature(self.audioAUG)
            import pandas as pd
            tmp = run_prediction(features)
            print tmp
            self.audioAUG, self.updateCount = [], 0
            self.predicted.pop()
            self.predicted.append('class')
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
            if np.max(self.ear.fft)>self.maxFFT:
                self.maxFFT=np.max(np.abs(self.ear.fft))
                #self.grFFT.plotItem.setRange(yRange=[0,self.maxFFT])
                self.grFFT.plotItem.setRange(yRange=[0,1])
            self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
            pen=pyqtgraph.mkPen(color='b')
            self.grPCM.addLegend(offset=(30, 30))
            className = self.accumlate(self.ear.datax)

            self.grPCM.plot(self.ear.datax,self.ear.data,pen=pen,clear = True, name = className)
            pen=pyqtgraph.mkPen(color='r')
            self.grFFT.plot(self.ear.fftx,self.ear.fft/self.maxFFT,pen=pen,clear=True)
        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()
    print("DONE")
