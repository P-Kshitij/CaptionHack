from keras.preprocessing import sequence 
from keras.preprocessing import text
import numpy as np
from keras.layers import LSTM, CuDNNLSTM
captions = [e.strip().split("\t") for e in open("../flickr/Flickr8k.token.txt").readlines()]
trainfiles = [e.strip() for e in open("../flickr/Flickr_8k.trainImages.txt").readlines()]
devfiles = [e.strip() for e in open("../flickr/Flickr_8k.devImages.txt").readlines()]
testfiles = [e.strip() for e in open("../flickr/Flickr_8k.testImages.txt").readlines()]
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

captionlist = []
for i,j in enumerate(traindict.values()):
    captionlist+=j

# phoolstop -> special word for full stop 
max_len=0
captioncleanlist = []
for i,k in enumerate(captionlist):
    if k[-1]=='.':
        k.append(.)
    clcaption=k.strip(' .').lower()
    captioncleanlist.append(clcaption+' phoolstop')
    max_len=max(max_len,len(captioncleanlist[i].split()))
len(captioncleanlist)

tok_caplist = [] #Tokenized Caption List
vocab = set() #The vocab set
for cap in captioncleanlist:
    tok_caplist.append(text.text_to_word_sequence(cap))
    for w in text.text_to_word_sequence(cap):
        vocab.add(w)

# Making an encoding dictionary
word_index=dict()
index_word=dict()
for i,w in enumerate(vocab):
    word_index[w]=i+1
    index_word[i+1]=w
word_index[None]=0
index_word[0]=None

cap_train = [tok_caplist[i] for i in range(0,30000,5)]

class Config:
    pass

config = Config()
config.max_len = max_len
config.vocab_size = len(word_index)
config.data_size = len(traindict)

enc_caps = []
for cap in cap_train:
    enc_cap = [word_index[word] for word in cap]
    enc_caps.append(enc_cap)

#Padding:
padd_caps=sequence.pad_sequences(sequences=enc_caps,maxlen=config.max_len,padding='post')


#One-encoding:
oneh_caps = np.zeros((config.data_size,config.max_len,config.vocab_size))
for i in range(config.data_size):
    for j in range(config.max_len):
        oneh_caps[i][j][padd_caps[i][j]]=1
print(oneh_caps.shape)

from keras.preprocessing import sequence 
from keras.preprocessing import text
import numpy as np

captions = [e.strip().split("\t") for e in open("../flickr/Flickr8k.token.txt").readlines()]
trainfiles = [e.strip() for e in open("../flickr/Flickr_8k.trainImages.txt").readlines()]
devfiles = [e.strip() for e in open("../flickr/Flickr_8k.devImages.txt").readlines()]
testfiles = [e.strip() for e in open("../flickr/Flickr_8k.testImages.txt").readlines()]
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

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
input_dir = '../flickr/Flicker8k_Dataset/'
img_height = 150
img_width = 150
img_chan= 3

train_imgs = []
for name in traindict:
    img = load_img(input_dir+name,target_size=(150,150))
    train_imgs.append(img_to_array(img))

dev_imgs = []
for name in devdict:
    img = load_img(input_dir+name,target_size=(150,150))
    dev_imgs.append(img_to_array(img))
dev_imgs = np.asarray(dev_imgs)
dev_imgs.shape

#conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))




# # imgmodel  = Sequential()
# # imgmodel.add(conv_base)
# # imgmodel.add(Flatten(input_shape=conv_base.output_shape[1:]))
# txt_lstm_model = Sequential()
# txt_lstm_model.add(Embedding(config.vocab_size, 100, input_length=config.maxlen)
# txt_lstm_model.add(CuDNNLSTM(config.hidden_dims, activation="relu")
# model=Sequential()
# model.add(imgmodel)
# model.add()

from keras.layers import BatchNormalization, Dense, RepeatVector

# The top layer is the last layer
image_model = VGG16(include_top=False, weights='imagenet', pooling='avg',input_shape=(150, 150, 3))
# Fix the weights
image_model.trainble=False

embedding_size = 300
dense_input = BatchNormalization(axis=-1)(image_model.output)
image_dense = Dense(units=embedding_size)(dense_input)  # FC layer
# Add a timestep dimension to match LSTM's input size
image_embedding = RepeatVector(1)(image_dense)
image_input = image_model.input

from keras.layers import Embedding, Input

config.embedding_size = 500
sentence_input = Input(shape=[None])
word_embedding = Embedding(input_dim=config.vocab_size,
                           output_dim=embedding_size
                           )(sentence_input)


from keras.layers import (BatchNormalization, Concatenate, Dense, LSTM,
                          TimeDistributed)
from keras.models import Model
from keras.optimizers import Adam

sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])

learning_rate = 0.00051
lstm_output_size = 500
#vocab_size = 2536
lstm_layers = 1
dropout_rate = 0.22
input_ = sequence_input

for _ in range(lstm_layers):
    input_ = BatchNormalization(axis=-1)(input_)
    lstm_out = LSTM(units=lstm_output_size,
                  return_sequences=True,
                  dropout=dropout_rate,
                  recurrent_dropout=dropout_rate)(input_)
    input_ = lstm_out

sequence_output = TimeDistributed(Dense(units=config.vocab_size))(lstm_out)

model = Model(inputs=[train_imgs, oneh_caps],
              outputs=sequence_output)
model.compile(optimizer=Adam(lr=learning_rate),
              loss=categorical_crossentropy_from_logits,
              metrics=[categorical_accuracy_with_variable_timestep])
