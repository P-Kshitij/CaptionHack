import os  
import shutil

os.mkdir('../flickr/Flicker8k_Dataset/train')
os.mkdir('../flickr/Flicker8k_Dataset/dev')
os.mkdir('../flickr/Flicker8k_Dataset/test')

train_dir = '../flickr/Flicker8k_Dataset/train/'
dev_dir = '../flickr/Flicker8k_Dataset/dev/'
test_dir = '../flickr/Flicker8k_Dataset/test/'


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

for img_name in traindict:
    shutil.move(input_dir+'train'+img_name, train_dir+img_name) 
    
for img_name in devdict:
    shutil.move(input_dir+'dev'+img_name, dev_dir+img_name) 
    
for img_name in testdict:
    shutil.move(input_dir+'test'+img_name, test_dir+img_name) 