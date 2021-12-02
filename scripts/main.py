from re import T
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import DataPreprocess
from networks import CNN_networks


class TrainModel(CNN_networks):
    def __init__(self) -> None:
        self.shape = (32, 32, 3)
        super().__init__(self.shape)
        self.data = DataPreprocess()
        
        self.net = CNN_networks(self.shape)
        self.num_classes = 43
        self.epochs = 10
        
        # self.model = self.loadmodel(True)
        
        pass


    def train(self):

        # load a blank model
        model = self.loadmodel(False)

        # load training and validation data
        x_train, y_train, s_train, c_train = self.data.load_data('train.pickle')
        x_valid, y_valid, s_valid, c_valid = self.data.load_data('valid.pickle')

        # normailize training and validataing dataset.
        x_train, x_valid = map(self.data.normalize_data, [x_train, x_valid])

        # shuffle dataset
        x_train, y_train = self.data.shuffle_data(x_train, y_train)
        x_valid, y_valid = self.data.shuffle_data(x_valid, y_valid)

        # one hot conversion
        y_train, y_valid = map(self.oneHot, [y_train, y_valid])

        hist = model.fit(x_train, y_train, epochs = self.epochs,
                validation_data = (x_valid, y_valid), verbose=1)

        print('Epochs={0:d}, training accuracy={1:.5f}, validation accuracy={2:.5f}'.\
        format(self.epochs, max(hist.history['accuracy']), max(hist.history['val_accuracy'])))

        # saving model
        model.save('model1')

        pass

    
    def test(self):

        model = self.loadmodel()
        # load test data
        x_test, y_test, s_test, c_test = self.data.load_data('test.pickle')

        # normalize test data
        x_test = self.data.normalize_data(x_test)

        # shuffle test data
        x_test, y_test = self.data.shuffle_data(x_test, y_test)

        # to categorial
        y_test = self.oneHot(y_test)

        model.evaluate(x_test, y_test)

        pass
    

    def oneHot(self, y):
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)

        return y

    def loadmodel(self, saved=True):
        if saved:
            try:
                model = keras.models.load_model('model')
                print('--- model loaded ---')
            except:
                model = self.net.CustomNet()
                print('--- model not loaded ---')
        else:
            model = self.net.CustomNet()
            print('--- model not loaded ---')
            
        return model


# tm = TrainModel()
# tm.test()

    

    
