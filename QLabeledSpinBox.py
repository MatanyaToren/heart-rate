import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSpinBox, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt



class QLabeledSpinBox(QWidget):
    def __init__(self, label='label for slider', range=(1,30), initValue:int = 20):
        super().__init__()
        
        self.SpinBox = QSpinBox()
        self.SpinBox.setFixedSize(70,40)
        font = self.SpinBox.font()
        font.setPointSize(12)
        self.SpinBox.setFont(font)
        self.SpinBox.setRange(*range)
        self.SpinBox.setValue(initValue)
        self.Label = QLabel()
        self.Label.setStyleSheet("""QLabel { 
                                    font-size : 10pt; 
                                    }""")
        self.Label.setText(label)
        self.Label.setAlignment(Qt.AlignCenter)
        
        self.Box = QVBoxLayout()
        self.Box.addWidget(self.Label, alignment=Qt.AlignCenter)
        self.Box.addWidget(self.SpinBox, alignment=Qt.AlignBottom | Qt.AlignCenter)
        
        self.setLayout(self.Box)
        
    def connect(self, changeValueFunc):
        self.SpinBox.valueChanged.connect(changeValueFunc)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QLabeledSpinBox()
    ex.show()
    sys.exit(app.exec_())