import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow import keras
from tensorflow.keras.callbacks import History, LearningRateScheduler 
from tensorflow.python.keras.utils.np_utils import to_categorical



def lr_change(epoch):
    lr = 0.0005
    if (epoch >= 10):
        lr = 0.0001
        if (epoch >= 20):
            lr = 0.00001
            if (epoch >= 40):
                lr = 0.000001
    return lr




# Read data from the CSV file
data = pd.read_csv('data_set.csv')
origin = data.copy()  
# print(origin.shape)

# id is not neccessary in the trainning, we use species name to by y
data.drop(columns=data.columns[0], axis=1, inplace=True)

# we have to trainsform the species name's into one hot version
# as we are using loss function categorical_crossentropy

size = 100

y = data.pop(str(size))
# print(y)
le = LabelEncoder().fit(y)
y = le.transform(y)

# change species names into one hot version
y_one_hot = to_categorical(y)
# print(y_one_hot)

# Standardising the data to give zero mean
X = data
le = StandardScaler().fit(data)
X = le.transform(data)

# create nn network
# x's size is 120 * 50
model = Sequential([
    Dense(512,input_dim=size,kernel_initializer = 'random_uniform', activation='relu'),
    Dense(256,activation='sigmoid'),
    # Dropout(0.3),
    # we need 99 outputs
    Dense(3, activation='softmax')
]) 

# model.summary()

# choose the model's parameters
opt = keras.optimizers.Adam(learning_rate=0.001)

lrate = LearningRateScheduler(lr_change)
callback_list = [lrate]
model.compile(optimizer='Adam', loss='MSE', metrics = ["accuracy"])
#sparese_categorical_crossentropy is used to do binary decisions


# history = model.fit(x = X, y = y_one_hot, batch_size=10,epochs=200,shuffle = True, 
#                     verbose=2,validation_split=0.2)

history = model.fit(x = X, y = y_one_hot, batch_size=10,epochs=100,shuffle = True, 
                    verbose=2,validation_split=0.2 ,callbacks=callback_list)




# output sample submission required by kaggle
# output = pd.DataFrame(y_pre,index=leaf_id,columns=sorted(origin.species.unique()))
# fp = open('nn_output.csv','w')
# fp.write(output.to_csv())