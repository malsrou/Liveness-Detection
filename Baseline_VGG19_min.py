from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten

# Create Baseline VGG Model and adapt to binary classification - Train only trailing dense layers and use the VGG feature extraction capabilities - Transfer Learning
def CreateBaselineModel():
  model = Sequential()
  model.add(VGG19(include_top=False, input_shape=(256, 256, 3)))

  for layer in model.layers:
    layer.trainable = False

  model.add(Flatten())
  model.add(Dense(1025, activation='relu'))#2560
  model.add(Dense(512, activation='relu'))#2560
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  return model
