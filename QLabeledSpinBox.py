import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSpinBox, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt



class QLabeledSpinBox(QWidget):
    def __init__(self, label='label for slider', range=(3,10), initValue:int =5):
        super().__init__()
        
        self.SpinBox = QSpinBox()
        self.SpinBox.setFixedSize(50,50)
        font = self.SpinBox.font()
        font.setPointSize(20)
        self.SpinBox.setFont(font)
        self.SpinBox.setValue(initValue)
        self.Label = QLabel()
        self.Label.setStyleSheet("""QLabel { 
                                    font-size : 12pt; 
                                    }""")
        self.Label.setText(label)
        self.Label.setAlignment(Qt.AlignCenter)
        
        self.Box = QVBoxLayout()
        self.Box.addWidget(self.Label, alignment=Qt.AlignCenter)
        self.Box.addWidget(self.SpinBox, alignment=Qt.AlignCenter)
        
        self.setLayout(self.Box)
        
    def connect(self, changeValueFunc):
        self.SpinBox.connect(changeValueFunc)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QLabeledSpinBox()
    ex.show()
    sys.exit(app.exec_())