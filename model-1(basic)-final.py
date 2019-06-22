from keras.preprocessing import sequence 
from keras.preprocessing import text
import numpy as np
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Embedding, Input
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Dense, RepeatVector, TimeDistributed
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam

input_dir = '../flickr/Flicker8k_Dataset/'
img_height = 150
img_width = 150
img_chan= 3
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
config.embedding_size = 300

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

# Loading the images:
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
#------------------------------------------------------------------

#--- The image embedding using VGG16---------------------------------------
# The top layer is the last layer
image_model = VGG16(include_top=False, weights='imagenet', pooling='avg',input_shape=(150, 150, 3))
# Fix the weights
image_model.trainable=False
dense_input = BatchNormalization(axis=-1)(image_model.output)
image_dense = Dense(units=embedding_size)(dense_input)  # FC layer
# Add a timestep dimension to match LSTM's input size
image_embedding = RepeatVector(1)(image_dense)
image_input = image_model.input
#---------------------------------------------------------------------------



sentence_input = Input(shape=[None])
word_embedding = Embedding(input_dim=config.vocab_size,
                           output_dim=config.embedding_size
                           )(sentence_input)



sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])

learning_rate = 0.00051
lstm_output_size = 500
lstm_layers = 1
dropout_rate = 0.22
input_ = sequence_input

import tensorflow as tf

def categorical_crossentropy_from_logits(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                 logits=y_pred)
    return loss

def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep

    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                              tf.argmax(y_pred, axis=1)),
                                    dtype=tf.float32))
    return accuracy

for _ in range(lstm_layers):
    input_ = BatchNormalization(axis=-1)(input_)
    lstm_out = CuDNNLSTM(units=lstm_output_size,
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

from keras_image_captioning.dataset_providers import DatasetProvider

dataset_provider = DatasetProvider()
epochs = 33  # the number of passes through the entire training set

# model is the same variable as the one in the previous code snippet
model.fit_generator(generator=dataset_provider.training_set(),
                    steps_per_epoch=dataset_provider.training_steps,
                    epochs=epochs,
                    validation_data=dataset_provider.validation_set(),
                    validation_steps=dataset_provider.validation_steps)

model.save("caption-gen1.h5")