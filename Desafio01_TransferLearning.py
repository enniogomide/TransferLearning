# Importação para execução do CNN - Transfer learning
import os
import requests
import zipfile
import shutil

import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

# ***************************************************************
# # helper function to load image 
# and return it and input vector
# ***************************************************************
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Parametros para treinar a rede em transfer-learning

PERCENTUAL_DA_AMOSTRA = 0.20

# *****************************************************************************
# *** Running in windows
# *****************************************************************************

# Images - download when running in windows

# url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
# response = requests.get(url)

# with open('kagglecatsanddogs_5340.zip', 'wb') as file:
#    file.write(response.content)

# with zipfile.ZipFile("kagglecatsanddogs_5340.zip","r") as zip_ref:
#     zip_ref.extractall(cwd)

# *****************************************************************************
# *** Running in colab
# *****************************************************************************

# images - download when running in colab
# IMAGES TO DOWNLOAD FOR THE TRANSFER LEARNING CHALLANGE 01
#!echo ""
#!curl -L -o 101_ObjectCategories.tar.gz --progress-bar https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
# !unzip 101_ObjectCategories.tar.gz

# To exclude files and folders not necessary in colab

# !rm 101_ObjectCategories.tar.gz
# #!rm redme[1].txt
# !rm CDLA-Permissive-2.0.pdf
# !ls



#
# Exclude files that will not be used anymore in windoes
#
cwd = os.getcwd()
fullPath = cwd + "\\kagglecatsanddogs_5340.zip"

if os.path.exists(fullPath):
    os.remove(cwd + "\\kagglecatsanddogs_5340.zip")
    os.remove(cwd + "\\readme[1].txt")
    os.remove(cwd + "\\CDLA-Permissive-2.0.pdf")

# *******************************************************
# CATEGORIES SELECTION FOR TRAINNING THE CNN
# *******************************************************
# At this cell we select the categories to be trainned, 
# excluding the ones we will not train the CNN.
#
root = 'PetImages'
exclude = []
train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(root) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

print(categories)

# Preparing folders for the training process with a smaller qty of images
# folder_Name = cwd + "\\Selected"
# if not os.path.exists(folder_Name):
#   os.makedirs(folder_Name)

# for sub_Folder in categories:
#   full_Sub_Folder = folder_Name + "\\" + sub_Folder
#   if os.path.exists(full_Sub_Folder):
#     shutil.rmtree(full_Sub_Folder)
#   os.makedirs(full_Sub_Folder)


# *********************************************************
# *** Load all images from root folder
# *********************************************************
data_to_exclude = []
data = []
for c, category in enumerate(categories):
    all_Images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    images = random.sample(all_Images, int(len(all_Images) * PERCENTUAL_DA_AMOSTRA))
    for img_path in images:
        try:
          img, x = get_image(img_path)
          data.append({'x':np.array(x[0]), 'y':c})
        except:
          data_to_exclude.append(img_path)

# count the number of classes
num_classes = len(categories)
print(num_classes)

# Randomize the data order
random.shuffle(data)

# create training / validation / test split (70%, 15%, 15%)

idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

# separate data for labels (deu erro aqui). Não foi
# possível alocar 2.1GB para o array

x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)

# Pre-process the data as before by making sure it's float32 
# and normalized between 0 and 1.

# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

# summary of what we have
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

#example of the data (images) to submit to the CNN
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)

#build a network from scratch with the data we have
# build the network
model = Sequential()
print("Input dimensions: ",x_train.shape[1:])

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# compile the model to use categorical cross-entropy loss function and adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_val, y_val))


# Validation loss and accuracy over time (plot)
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

#Final evaluation by running the model on the training set
# results

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# **********************************************************
# **** Transfer learning by starting with existing network
# **********************************************************

vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

#
# How it is done: Remove the final classification layer 1000 neuron softmax at the end,
# that correspond to ImageNet, and replace it with a new softmax layer for our dataset, which contains 2 neurons
# in implementation, we create a copy of VGG from its input layer, until de second to
# to the last one, and work with that. instead of modifying.
# To do that, use de keras model.

# make a reference to VGG's input layer
inp = vgg.input

# make a new softmax layer with num_classes neurons
new_classification_layer = Dense(num_classes, activation='softmax')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(vgg.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)

# retrain the network.
# model_new on the new dataset adn labels
# first: need to freeze the weights and biases in all the layer in the network,
# except ou new one at the end.
# need to set the trainable flag to false.

# make all layers untrainable by freezing weights (except for last layer)
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

# ensure the last layer is trainable/not frozen
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

#Training with transfer learning

history2 = model_new.fit(x_train, y_train,
                         batch_size=128,
                         epochs=10,
                         validation_data=(x_val, y_val))

# validation plot (loss and accuracy)
# blue - scratch training
# green - the new transfer learned

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.plot(history2.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()