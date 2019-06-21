from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
captions = [e.strip().split("\t") for e in open("./flickr/Flickr8k.token.txt").readlines()]
trainfiles = [e.strip() for e in open("./flickr/Flickr_8k.trainImages.txt").readlines()]
devfiles = [e.strip() for e in open("./flickr/Flickr_8k.devImages.txt").readlines()]
testfiles = [e.strip() for e in open("./flickr/Flickr_8k.testImages.txt").readlines()]
from collections import defaultdict
captiondict = defaultdict(list)
for pair in captions:
    image = pair[0].split('#')[0]
    caption = pair[1]
    captiondict[image].append(caption)
traindict = {}
devdict = {}
testdict = {}
for f in trainfiles:
    traindict[f] = captiondict[f]
for f in devfiles:
    devdict[f] = captiondict[f]
for f in testfiles:
    testdict[f] = captiondict[f]