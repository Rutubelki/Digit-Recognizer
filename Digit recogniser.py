#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf


# #using mnsit dataset

# In[21]:


mnist=tf.keras.datasets.mnist


# In[22]:


import time

start_time = time.time()
(x_train,y_train),(x_test,y_test)=mnist.load_data()
end_time = time.time()

print(f"Time taken to load the MNIST dataset: {end_time - start_time:.2f} seconds")


# In[23]:


x_train.shape


# In[24]:


print(x_train[0])


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


plt.imshow(x_train[0])
plt.show()


# In[27]:


plt.imshow(x_train[0],cmap=plt.cm.binary)


# In[28]:


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)


# In[29]:


print(x_train[0])


# In[30]:


print(y_train[0])


# In[31]:


import numpy as np


# In[32]:


Img_size=28
x_trainr=np.array(x_train).reshape(-1,Img_size,Img_size,1)
x_testr=np.array(x_test).reshape(-1,Img_size,Img_size,1)
print(x_trainr.shape)
print(x_testr.shape)


# In[33]:


print(x_trainr.shape[1:])


# In[34]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D


# In[35]:


model=Sequential()

#First convolution layer
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform',input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Second convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Third convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

#Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

#Last Fully connected layer 2
model.add(Dense(10))
model.add(Activation("softmax"))


# In[36]:


model.summary()


# In[37]:


print("Total number of samples",len(x_trainr))


# In[ ]:





# In[38]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


# In[ ]:





# In[39]:


model.fit(x_trainr,y_train,epochs=5,validation_split=0.3,batch_size=1)


# In[40]:


test_loss,test_acc=model.evaluate(x_testr,y_test,batch_size=1)
print("Test loss",test_loss)
print("Test_accuracy",test_acc)


# In[41]:


predictions=model.predict([x_testr])


# In[42]:


print(predictions)


# In[45]:


print(np.argmax(predictions[0]))


# In[46]:


plt.imshow(x_test[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




