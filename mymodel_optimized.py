from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.backend import dropout

def CreateMyModel():

    in_layer = layers.Input((64,64,3))
    conv1 = layers.Conv2D(4, kernel_size=(17,17), strides=1, kernel_initializer='uniform', kernel_regularizer='l2', activation='relu')(in_layer)
    maxp = layers.MaxPool2D((2,2), strides=1)(conv1)
    conv2 = layers.Conv2D(8, kernel_size=(5,5), kernel_initializer='uniform', kernel_regularizer='l2', activation='relu')(maxp)
    maxp1 = layers.MaxPool2D((2,2), strides=1)(conv2)
    conv3 = layers.Conv2D(16, kernel_size=(3,3), kernel_regularizer='l2')(maxp1)
    flattened = layers.Flatten()(conv3)
    dense1 = layers.Dense(1024, activation='relu')(flattened)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    dense3 = layers.Dense(50, activation='relu')(dense2)
    preds = layers.Dense(1, activation='sigmoid')(dense3)

    model = Model(in_layer, preds)

    return model



"""""
in_layer = layers.Input((64,64,3))
    conv1 = layers.Conv2D(64, kernel_size=(7,7), kernel_initializer='uniform', kernel_regularizer='l2')(in_layer)
    lr1= layers.LeakyReLU(alpha=0.2)(conv1)
    batchn = layers.BatchNormalization()(lr1)
    drop0 = layers.Dropout(0.5)(batchn)
    conv2 = layers.Conv2D(256, kernel_size=(7,7))(drop0)
    lr2= layers.LeakyReLU(alpha=0.1)(conv2)
    batchn1 = layers.BatchNormalization()(lr2)
    drop1 = layers.Dropout(0.5)(batchn1)
    conv3 = layers.Conv2D(128, kernel_size=(5,5))(drop1)
    lr3= layers.LeakyReLU(alpha=0.1)(conv3)
    pool1 = layers.AveragePooling2D(2,2)(lr3)
    batchn2 = layers.BatchNormalization()(pool1)
    flattened = layers.Flatten()(batchn2)
    dense1 = layers.Dense(1024)(flattened)
    lr4= layers.LeakyReLU(alpha=0.1)(dense1)
    drop2 = layers.Dropout(0.8)(lr4)
    dense2 = layers.Dense(512)(drop2)
    lr5= layers.LeakyReLU(alpha=0.1)(dense2)
    dense3 = layers.Dense(10)(drop2)
    preds = layers.Dense(1, activation='sigmoid')(dense3)

    model = Model(in_layer, preds)
    """""