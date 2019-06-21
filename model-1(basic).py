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



