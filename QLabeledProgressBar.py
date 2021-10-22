import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt



class QLabeledProgressBar(QWidget):
    def __init__(self, label='label for slider', range=(2,30), colormap={'red': (2,20), 'green': (20,30)}, *args, **kwargs):
        super().__init__()
        
        self.colormap = colormap
        self.ProgressBar = QProgressBar(*args, **kwargs)
        self.ProgressBar.setOrientation(Qt.Vertical)
        self.ProgressBar.setAlignment(Qt.AlignCenter)
        self.ProgressBar.setRange(*range)
        self.setValue(20)

        self.Label = QLabel()
        self.Label.setText(label)
        self.Label.setAlignment(Qt.AlignCenter)
        
        self.Box = QVBoxLayout()
        self.Box.addWidget(self.ProgressBar, alignment=Qt.AlignCenter)
        self.Box.addWidget(self.Label)
        
        self.setLayout(self.Box)
        
        
    def setValue(self, value):
        for key, (min, max) in self.colormap.items():
            if value >= min and value < max:
                self.setColor(key)
                self.ProgressBar.setValue(value)
                self.ProgressBar.setFormat('{}'.format(value))
            
    def setColor(self, color):
        self.ProgressBar.setStyleSheet(
                    """QProgressBar::chunk{
                    background-color: """ + color + """;
                    }""")
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QLabeledProgressBar(label='snr\n[dB]')
    ex.setValue(10)
    ex.show()
    sys.exit(app.exec_())