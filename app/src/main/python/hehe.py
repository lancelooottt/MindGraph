from os.path import dirname, join

import pandas as pd


def output():
    filename = join(dirname(__file__),"text_emotion.csv")
    data = pd.read_csv(filename)
    shape = data.shape

    return shape

#data = pd.read_csv('text_emotion.csv')
#shape = data.shape
#print(shape)
