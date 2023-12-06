#%%
import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
#%%
data_dir = 'C:/Users/Msi/Desktop/CALISMALAR/Money Detection/data'
image_exts = ['jpeg','jpg', 'bmp', 'png']
#%%


#%%
data = tf.keras.utils.image_dataset_from_directory('data')
#%%
data_iterator = data.as_numpy_iterator()

#%%
batch = data_iterator.next()
#%%
batch[1]

#%%
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(str(batch[1][idx]))  # Convert label to string

plt.show()
#%%
def scale_and_normalize(x, y):
    x_scaled = x / 255.0
    max_value = tf.reduce_max(x_scaled)
    x_normalized = x_scaled / max_value
    return x_normalized, y
data = data.map(scale_and_normalize)

# Check the maximum value in the first batch
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
print(batch[0].max())

#%%
#%%
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
#%%
train_size
#%%
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

print(train_size)
print(val_size)
print(test_size)

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
#%%
model = Sequential()

#%%
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#%%
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
#%%
model.summary()
#%%
logdir='logs'
#%%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#%%
hist = model.fit(train, epochs=3, validation_data=val, callbacks=[tensorboard_callback])
 #%%
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#%%
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#%%
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
#%%
for batch in train.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
#%%
print(pre.result(), re.result(), acc.result())

#%%
img = cv2.imread('5tl2.jpg')
plt.imshow(img)
plt.show()

#%%
resize = tf.image.resize(img, (255,255))
plt.imshow(resize.numpy().astype(int))
plt.show()

#%%
yhat = model.predict(np.expand_dims(resize/255, 0))
yhat
print(yhat)
#%%
if yhat > 0.5:
    print(f'Predicted class is 5 TL')
else:
    print(f'Predicted class is 10 TL')
#%%



