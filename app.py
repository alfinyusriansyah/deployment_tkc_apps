# Import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable
import streamlit as st
import numpy as np
from numpy import linalg as LA
import torchvision
import torch.nn.functional as F
import time
import os
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
import os
warnings.filterwarnings('ignore')

# Create the network to extract the features
class MyResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, transform_input=False):
        super(MyResNetFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.fc = resnet.fc
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.conv1(x)
        # 149 x 149 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # 147 x 147 x 32
        x = self.layer1(x)
        # 147 x 147 x 64
        x = self.layer2(x)
        # 73 x 73 x 64
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=7)

        return x
# Import pre-trained model from by using torchvision package
model = torchvision.models.resnet50(pretrained = True) # resnet 50 model is imported

# set the model train False since we are using our feature extraction network 
model.train(False)

# Set our model with pre-trained model 
my_resnet = MyResNetFeatureExtractor(model)

def extractor(data):
        since = time.time()
        # read images images from a directory
        list_imgs_names = os.listdir(data)
        #list_imgs_names
        # create an array to store features 
        N = len(list_imgs_names)
        fea_all = np.zeros((N, 2048))
        # define empy array to store image names
        image_all = []
        # extract features 
        for ind, img_name in enumerate(list_imgs_names):
            img_path = os.path.join(data, img_name)
            image_np = Image.open(img_path)
            image_np = np.array(image_np)
            image_np = resize(image_np, (224, 224))
            image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
            image_np = Variable(image_np.unsqueeze(0))   #bs, c, h, w
            fea = my_resnet(image_np)
            fea = fea.squeeze()
            fea = fea.cpu().data.numpy()
            fea = fea.reshape((1, 2048))
            fea = fea / LA.norm(fea)
            fea_all[ind] = fea
            image_all.append(img_name)

        time_elapsed = time.time() - since 

        st.write('Feature extraction complete in {:.02f}s'.format(time_elapsed % 60))

        return fea_all, image_all

path = './dataset'
feats, image_list = extractor(path)

def load_image(image_file):
	img = Image.open(image_file)
	return img

image_file = st.file_uploader("Choose a image file", type=['png','jpeg','jpg'])

if image_file is not None:
    img = load_image(image_file)
    with open(os.path.join("test",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    image = Image.open(image_file)
    st.image(image, caption='QUERY Image. Source from Google')
    st.success("Saved File to Directory test")

    Genrate_pred = st.button("Generate Retrieval")    
    if Genrate_pred:
        # test image path
        test = './test'
        feat_single, image = extractor(test)

        scores  = np.dot(feat_single, feats.T)
        sort_ind = np.argsort(scores)[0][::-1]
        scores = scores[0, sort_ind]

        maxres = 20
        imlist = [image_list[index] for i, index in enumerate(sort_ind[0:maxres])]
        st.write ("top %d images in order are: " %maxres, imlist)

        fig=plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('xkcd:white')

        for i in range(len(imlist)):
            sample = imlist[i]
            img = mpimg.imread('./dataset' + '/' + sample)
            #ax = plt.subplot(figsize)
            ax = fig.add_subplot(4, 5, i+1)
            ax.autoscale()
            plt.tight_layout()
            plt.imshow(img, interpolation='nearest')
            ax.set_title('{:.3f}%'.format(scores[i]))
            ax.axis('off')
        st.pyplot(fig)
        
        dir = './test'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))