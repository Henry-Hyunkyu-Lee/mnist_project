import tensorflow as tf

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.i = 0
        self.model = None
        self.data = None

        self.datas = QComboBox()
        self.datas.addItem('No Data')
        self.datas.addItem('mnist')
        self.datas.addItem('fashion_mnist')
        self.datas.activated[str].connect(self.select_data)

        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.labelLabel = QLabel()
        self.labelLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.update_pixmap(i=self.i)
        
        self.p_btn = QPushButton('prev')
        self.p_btn.clicked.connect(self.prev_img)
        self.n_btn = QPushButton('next')
        self.n_btn.clicked.connect(self.next_img)

        self.option = QComboBox()
        self.option.addItem('No Model')
        self.option.addItem('Multi Layer Perceptron')
        self.option.addItem('Conv Net')
        self.option.activated[str].connect(self.select_model)

        self.headerLabel = QLabel('Pred')
        self.answerLabel = QLabel()
        self.headerLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.answerLabel.setAlignment(QtCore.Qt.AlignCenter)


        btn_layout = QGridLayout()
        btn_layout.addWidget(self.p_btn, 0, 0)
        btn_layout.addWidget(self.n_btn, 0, 1)


        layout = QGridLayout()
        layout.addWidget(self.datas, 0, 0)
        layout.addWidget(self.imageLabel, 1, 0)
        layout.addWidget(self.labelLabel, 2, 0)
        layout.addLayout(btn_layout, 3, 0)
        layout.addWidget(self.option, 4, 0)
        layout.addWidget(self.headerLabel, 5, 0)
        layout.addWidget(self.answerLabel, 6, 0)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.show()

    def update_pixmap(self, i=0):
        img = self.x_test[i]
        qimg = QImage(img.data, 28, 28, img.strides[0], QImage.Format_Indexed8)
        pxmap = QPixmap.fromImage(qimg)
        scaled = pxmap.scaled(80,80)
        self.imageLabel.setPixmap(scaled)

        label = self.y_test[i]
        label = str(label)
        self.labelLabel.setText(label)

        if self.model:
            inp = tf.cast(img, tf.float32) / 255.0
            inp = tf.expand_dims(inp, axis=0)
            predict = self.model.predict(inp)
            predict = int(tf.argmax(predict[0]))
            self.answerLabel.setText('{}'.format(predict))


    def prev_img(self):
        if self.data:
            self.i -= 1
            self.update_pixmap(self.i)
        else:
            print('No data, Please select dataset.')

    def next_img(self):
        if self.data:
            self.i += 1
            self.update_pixmap(self.i)
        else:
            print('No data, Please select dataset.')

    def select_data(self, text):
        if text == 'No Data':
            x_train, y_train, x_test, y_test = None, None, None, None
            self.data = None
        if text == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if text == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data = text

    def select_model(self, text):
        models = {
            'No Model': None,
            'Multi Layer Perceptron': 'mlp.h5',
            'Conv Net': 'convnet.h5'
        }
        if models[text]:
            if self.data == 'mnist':
                self.model = tf.keras.models.load_model('models/{}'.format(models[text]))
            elif self.data == 'fashion_mnist':
                self.model = tf.keras.models.load_model('models/fashion_{}'.format(models[text]))
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