# https://www.kaggle.com/valentynsichkar/traffic-signs-classification-with-cnn/data?select=datasets_preparing.py
# https://youtu.be/YXPyB4XeYLA?t=6623
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



class DataPreprocess():
    def __init__(self):
        self.file_path = os.path.join(os.getcwd(), 'input/')
        self.label_list = self.labelname_list('label_names.csv')
        pass

    def load_data(self, file):
        file_path = self.file_path + str(file)
        with open(file_path, 'rb') as f:
            d = pkl.load(f, encoding = 'latin1')
            x = d['features'].astype(np.float32)
            y = d['labels']
            c = d['coords']
            s = d['sizes']

        return x, y, s, c

    def shuffle_data(self, x, y):
        x, y = shuffle(x, y, random_state=0)

        return x, y

    def normalize_data(self, x):
        x = x/255.0

        return x


    def labelname_list(self, file='label_names.csv'):
        file_path = self.file_path + str(file)
        labels = pd.read_csv(file_path)
        label_list = []
        # print(len(labels))
        for row in range(len(labels)):
            label_list.append(labels['SignName'][row])
        
        return label_list



    def show_image(self, x_i, y_i):
        
        plt.figure()
        plt.imshow(x_i)

        plt.title(self.label_list[y_i])
        plt.show()



