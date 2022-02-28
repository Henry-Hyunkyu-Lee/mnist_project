import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.i = 0
        self.model = None

        self.imageLabel = QLabel()
        self.labelLabel = QLabel()
        self.labelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.update_pixmap(i=self.i)
        
        self.p_btn = QPushButton('prev')
        self.p_btn.clicked.connect(self.prev_img)
        self.n_btn = QPushButton('next')
        self.n_btn.clicked.connect(self.next_img)

        self.option = QComboBox()
        self.option.addItem('No Model')
        self.option.addItem('Multi Layer Perceptron')
        self.option.addItem('Conv Net')
        self.option.activated[str].connect(self.select_model)

        self.answerLabel = QLabel()
        self.answerLabel.setAlignment(QtCore.Qt.AlignCenter)


        btn_layout = QGridLayout()
        btn_layout.addWidget(self.p_btn, 0, 0)
        btn_layout.addWidget(self.n_btn, 0, 1)


        layout = QGridLayout()
        layout.addWidget(self.imageLabel, 0, 0)
        layout.addWidget(self.labelLabel, 1, 0)
        layout.addLayout(btn_layout, 2, 0)
        layout.addWidget(self.option, 3, 0)
        layout.addWidget(self.answerLabel, 4, 0)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.show()

    def update_pixmap(self, i=0):
        img = x_test[i]
        qimg = QImage(img.data, 28, 28, img.strides[0], QImage.Format_Indexed8)
        pxmap = QPixmap.fromImage(qimg)
        scaled = pxmap.scaled(240,240)
        self.imageLabel.setPixmap(scaled)

        label = y_test[i]
        label = str(label)
        self.labelLabel.setText(label)

        if self.model:
            inp = tf.cast(img, tf.float32) / 255.0
            inp = tf.expand_dims(inp, axis=0)
            predict = self.model.predict(inp)
            predict = str(tf.argmax(predict[0]))
            self.answerLabel.setText(predict)


    def prev_img(self):
        self.i -= 1
        self.update_pixmap(self.i)

    def next_img(self):
        self.i += 1
        self.update_pixmap(self.i)

    def select_model(self, text):
        models = {
            'No Model': None,
            'Multi Layer Perceptron': 'mlp.h5',
            'Conv Net': 'convnet.h5'
        }
        if models[text]:
            self.model = tf.keras.models.load_model('models/{}'.format(models[text]))
            print('Well loaded {}'.format(models[text]))
        else:
            self.model = None
            print('No model loaded.')
        self.update_pixmap(self.i)
        

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    window = ImageViewer()

    app.exec_()