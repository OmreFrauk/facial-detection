import cv2 
import os 
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import load_img, img_to_array

picture_size = 48
images = "./images"

def load_images_from_folder(expression):
    plt.figure(figsize=[12,12])    
    for i in range(1,10,1):
        plt.subplot(3,3,i)
        img = cv2.imread(os.path.join(f'./images/train/{expression}',random.choice(os.listdir(f'./images/train/{expression}'))))
        plt.imshow(img)
    plt.show()




