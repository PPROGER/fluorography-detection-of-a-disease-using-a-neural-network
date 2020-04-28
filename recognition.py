#!/usr/bin/python3

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# Размер изображения
image_size = 299
# Размер мини-выборки
batch_size = 32


    
model = load_model("inceptionv3_fine_tuned.h5")

#img_path = 'base_dir/val_dir/Tuberculosis/CHNCXR_0337_1.png'
img_path = '/home/pproger/Desktop/fluorography detection of a disease using a neural network/val_dir/Normal/CHNCXR_0171_0.png'
#img_path = 'base_dir/val_dir/Tuberculosis/MCUCXR_0367_1.png'
img = image.load_img(img_path, target_size=(image_size, image_size))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)

print(prediction)

if prediction[[0]] < 0.5:
    print('Normal')
else:
    print('Tuberculosis')
