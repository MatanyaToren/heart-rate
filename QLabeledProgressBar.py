import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont



class QLabeledProgressBar(QWidget):
    def __init__(self, label='label for slider', format='{:.1f}', range=(2,30), 
                 colormap={'red': (2,20), 'green': (20,30)}, 
                 printMessage=None, *args, **kwargs):
        super().__init__()
        
        self.colormap = colormap
        self.printMessage = printMessage
        self.name = kwargs['objectName']
        self.format = format
        self.ProgressBar = QProgressBar(*args, **kwargs)
        self.ProgressBar.setOrientation(Qt.Vertical)
        self.ProgressBar.setAlignment(Qt.AlignCenter)
        self.ProgressBar.setRange(100*range[0], 100*range[1])
        self.ProgressBar.setFixedSize(40,100)
        self.ProgressBar.setFont(QFont('Arial', 10))
        self.setValue(0)

        self.Label = QLabel()
        self.Label.setText(label)
        self.Label.setStyleSheet("""QLabel { 
                                   color : black;
                                   font-size : 12pt; 
                                   }""")
        self.Label.setAlignment(Qt.AlignCenter)
        
        self.Box = QVBoxLayout()
        self.Box.addWidget(self.ProgressBar, alignment=Qt.AlignCenter)
        self.Box.addWidget(self.Label)
        
        self.setLayout(self.Box)
        
        
    def setValue(self, value):
        self.ProgressBar.setValue(int(100*value))
        self.ProgressBar.setFormat(self.format.format(value))
        
        for key, (min, max) in self.colormap.items():
            if value >= min and value < max:
                self.setColor(key)
                
        if self.isValueRed(value) and self.printMessage is not None:
            self.printMessage(self.name, flag=True)
        
        elif self.printMessage is not None:
            self.printMessage(self.name, flag=False)
            
    def setColor(self, color):
        self.ProgressBar.setStyleSheet(
                    """QProgressBar::chunk{
                    background-color: """ + color + """;
                    }""")
        
    def isValueRed(self, value):
        try:
            range = self.colormap['red']
        except KeyError:
            return False
        
        return (value >= range[0] and value < range[1])
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QLabeledProgressBar(label='snr\n[dB]')
    ex.setValue(10)
    ex.show()
    sys.exit(app.exec_())