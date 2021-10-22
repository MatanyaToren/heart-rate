import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSpinBox, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt



class QLabeledSpinBox(QWidget):
    def __init__(self, label='label for slider'):
        super().__init__()
        
        self.SpinBox = QSpinBox()
        self.Label = QLabel()
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