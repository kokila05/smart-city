#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111')


# In[3]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[2]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[5]:


get_ipython().system('pip install matplotlib')
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[3]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model


# In[7]:


img = 'https://ultralytis.com/image/zidane.jpg'


# In[8]:


results = model(img)
results.print()


# In[1]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    

    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[10]:


import uuid   
import os
import time


# In[27]:


IMAGES_PATH = os.path.join('data', 'images') #/data/images
labels = ['human', 'object']
number_imgs = 5


# In[9]:


for label in labels:
    print('collecting images for {}'.format(label))
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
    imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
    print(imgname)


# In[29]:


cap = cv2.VideoCapture(0)
# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        # Webcam feed
        ret, frame = cap.read()
        
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        # Writes out image to file 
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[17]:


print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))


# In[26]:


get_ipython().system('git clone https://github.com/tzutalin/labelImg')


# In[ ]:




