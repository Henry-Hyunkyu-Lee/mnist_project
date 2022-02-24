import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPalette, qRgb
from PyQt5 import QtCore

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.i = 0

        self.imageLabel = QLabel()
        self.labelLabel = QLabel()
        self.labelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.update_pixmap(i=self.i)

        self.btn = QPushButton('next')
        self.btn.clicked.connect(self.next_img)

        layout = QGridLayout()
        layout.addWidget(self.imageLabel, 0, 0)
        layout.addWidget(self.labelLabel, 1, 0)
        layout.addWidget(self.btn, 2, 0)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.show()

    def update_pixmap(self, i=0):
        img = x_train[i]
        qimg = QImage(img.data, 28, 28, img.strides[0], QImage.Format_Indexed8)
        pxmap = QPixmap.fromImage(qimg)
        scaled = pxmap.scaled(280,280)
        self.imageLabel.setPixmap(scaled)

        label = y_train[i]
        label = str(label)
        self.labelLabel.setText(label)

    def next_img(self):
        self.i += 1
        self.update_pixmap(self.i)

        

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    window = ImageViewer()

    app.exec_()