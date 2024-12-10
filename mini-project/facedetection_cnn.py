#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import glob

#Specifying he directory of the dataset
Image_Dir = r"E:\Sem 7\IVP\Mini Project\Python-1\UTKFace"
Train_Size = 0.7
I_W = I_H = 198

Gender = {0: 'male', 1: 'female'}
Gender_Map = dict((g, i) for i, g in Gender.items())
Race = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
Race_Map = dict((r, i) for i, r in Race.items())

def parse_filepath(filepath):
    ''' For Extracting Images in the mentioned Directory'''
    global Gender, Race
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), Gender[int(gender)], Race[int(race)]
    except Exception as e:
        print(filepath, e)
        return None, None, None

files = glob.glob(os.path.join(Image_Dir, "*.jpg"))

attributes = list(map(parse_filepath, files))

#Making a dataframe with images as input and corresponding features as output.
df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['age', 'gender', 'race', 'file']
df = df.dropna()
df.head()

df.groupby(by=['race', 'gender'])['age'].count().plot(kind='bar')

#Making a permutation of the dataset and splitting into training and test set.
Len = len(df)
Permutation = np.random.permutation(Len)
Training_Size = int((Len)*(Train_Size))
Training_Set = Permutation[:Training_Size]
Test_Set = Permutation[Training_Size:]

Training_Size = int(Train_Size * 0.7)
Training_Set, Validation_Set = Training_Set[:Training_Size], Training_Set[Training_Size:]

df['gender_id'] = df['gender'].map(lambda gender: Gender_Map[gender])
df['race_id'] = df['race'].map(lambda race: Race_Map[race])

Max_Age = df['age'].max()

from keras.utils import to_categorical
from PIL import Image

def get_data_generator(df, indices, for_training, batch_size=16):
    '''Normalizing the image pixels and converting into categorical variables.'''
    global I_W, I_H, Max_Age
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((I_W, I_H))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / Max_Age)
            races.append(to_categorical(race, len(Race_Map)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break

#Importing libraries necessary to build the CNN
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D
from keras.optimizers import SGD
from keras.models import Model

#Specifying the sizeof the input images so that they can be reshaped to the specified size.
input_layer = Input(shape=(I_H, I_W, 3))

#Convolution Layers followed by a max pooling
C1 = Conv2D(32, kernel_size = (3,3), strides = (1,1), activation = 'relu')(input_layer)
M1 = MaxPool2D(pool_size=(2,2), strides=(2,2))(C1)

C2 = Conv2D(32*2, (3,3), activation = 'relu')(M1) 
M2 = MaxPool2D(pool_size=(2,2))(C2) 

C3 = Conv2D(32*4, (3,3), activation = 'relu')(M2) 
M3 = MaxPool2D(pool_size=(2,2))(C3) 

C4 = Conv2D(32*4, (3,3), activation = 'relu')(M3) 
M4 = MaxPool2D(pool_size=(2,2))(C4) 

C4 = Conv2D(32*5, (3,3), activation = 'relu')(M3) 
M4 = MaxPool2D(pool_size=(2,2))(C4) 

C5 = Conv2D(32*6, (3,3), activation = 'relu')(M3) 
M4 = MaxPool2D(pool_size=(2,2))(C5) 

bottleneck = GlobalMaxPool2D()(M4)

#Creating output layers
C = Dense(units=128, activation='relu')(bottleneck)
age_output = Dense(units=1, activation='sigmoid', name='age_output')(C)

C = Dense(units=128, activation='relu')(bottleneck)
race_output = Dense(units=len(Race_Map), activation='softmax', name='race_output')(C)

C = Dense(units=128, activation='relu')(bottleneck)
gender_output = Dense(units=len(Gender_Map), activation='softmax', name='gender_output')(C)

model = Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])

#Compiling our model with adam optimizer.
model.compile(optimizer='adam', 
              loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
              loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
              metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'})

#Extracting training and test sets from the dataframe.
batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, Training_Set, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, Validation_Set, for_training=True, batch_size=valid_batch_size)

from keras.callbacks import Callback

class MyLogger(Callback):
    '''For logging during each epoch'''
    def on_epoch_end(self, epoch, logs=None):
        with open('log.txt', 'a+') as f:
            f.write('%02d %.3f\n' % (epoch, logs['loss']))

mylogger = MyLogger()

#Training our model upon training set and applying the correlations on validation set.
output = model.fit(train_gen,
                    steps_per_epoch=len(Training_Set)//batch_size,
                    epochs=20,
                    verbose = 1,
                    validation_data=valid_gen,
                    validation_steps=len(Validation_Set)//valid_batch_size)

test_gen = get_data_generator(df, Test_Set, for_training=False, batch_size=128)
dict(zip(model.metrics_names, model.evaluate_generator(test_gen, steps=len(Test_Set)//128)))

test_gen = get_data_generator(df, Test_Set, for_training=False, batch_size=128)
x_test, (age_true, race_true, gender_true)= next(test_gen)
age_pred, race_pred, gender_pred = model.predict_on_batch(x_test)

race_true, gender_true = race_true.argmax(axis=-1), gender_true.argmax(axis=-1)
race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)
age_true = age_true * Max_Age
age_pred = age_pred * Max_Age

#For getting the accuracies achiened dur
from sklearn.metrics import classification_report
print("Classification report for race")
print(classification_report(race_true, race_pred))

print("\nClassification report for gender")
print(classification_report(gender_true, gender_pred))

#Applying our correlations on some of the images for comparing actual values with predicted values
import math
n = 30
random_indices = np.random.permutation(n)
n_cols = 5
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))

for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(x_test[img_idx])
    ax.set_title('a:{}, g:{}, r:{}'.format(int(age_pred[img_idx]), Gender[gender_pred[img_idx]], Race[race_pred[img_idx]]))
    ax.set_xlabel('a:{}, g:{}, r:{}'.format(int(age_true[img_idx]), Gender[gender_true[img_idx]], Race[race_true[img_idx]]))
    ax.set_xticks([])
    ax.set_yticks([])
