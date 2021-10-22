import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt



class QLabeledProgressBar(QWidget):
    def __init__(self, label='label for slider', *args, **kwargs):
        super().__init__()
        
        self.ProgressBar = QProgressBar(*args, **kwargs)
        self.ProgressBar.setOrientation(Qt.Vertical)
        self.ProgressBar.setAlignment(Qt.AlignCenter)
        # self.ProgressBar.setGeometry(0,0,20,30)
        self.ProgressBar.setValue(20)
        self.ProgressBar.setStyleSheet("""QProgressBar::chunk
                    {
                    background-color: green;
                    }""")
        # self.snrLevelBar.setTextVisible(False)
        self.ProgressBar.setFormat('hi')
        self.Label = QLabel()
        self.Label.setText(label)
        self.Label.setAlignment(Qt.AlignCenter)
        
        self.Box = QVBoxLayout()
        self.Box.addWidget(self.ProgressBar, alignment=Qt.AlignCenter)
        self.Box.addWidget(self.Label)
        
        self.setLayout(self.Box)
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QLabeledProgressBar()
    ex.show()
    sys.exit(app.exec_())