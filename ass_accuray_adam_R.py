import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow import keras
from tensorflow.keras.callbacks import History 
from tensorflow.python.keras.utils.np_utils import to_categorical

#for drawing
import matplotlib.pyplot as plt

# from keras.utils.np_utils import to_categorical

# Read data from the CSV file
data = pd.read_csv('./input/train.csv')
origin = data.copy()  
print(origin.shape)

# id is not neccessary in the trainning, we use species name to by y
data.drop(columns=data.columns[0], axis=1, inplace=True)

# print(data.shape)

## we have to trainsform the species name's into one hot version
y = data.pop('species')

# y = LabelEncoder().fit(y).transform(y)

# Onehot needs 2D array
le = LabelEncoder().fit(y)
y = le.transform(y)

# change id's into one hot version
y_one_hot = to_categorical(y)
print(y_one_hot.shape)


# Standardising the data to give zero mean
le = StandardScaler().fit(data)
X = le.transform(data)
# print(X.shape)


loss = []

ways = ['Adam','RMSprop']
# create nn network
# x's size is 990 * 192
for i in ways:
    model = Sequential([
        Dense(512,input_dim=192,kernel_initializer = 'random_uniform', activation='relu'),
        #Dropout(0.3),
        # we need 99 outputs
        Dense(99, activation='softmax')
    ]) 

    # model.summary()

    # choose the model's parameters
#     opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=i, loss='categorical_crossentropy', metrics = ["accuracy"])
    #sparese_categorical_crossentropy???

    # history = model.fit(x = X, y = y_one_hot, batch_size=32,epochs=400,shuffle = True, verbose=0)
    history = model.fit(x = X, y = y_one_hot, batch_size=64,epochs=400,shuffle = True, verbose=0,validation_split=0.2)
    # print(history.history['val_loss'])
    loss.append(history.history['accuracy'])

x = np.arange(1,401)
color = ['y','b','g','c','m','r']

for i in range(0,2):
    plt.plot(x,loss[i],color = color[i],label = ways[i])
plt.xlabel("epochs")
plt.ylabel("val_loss")
plt.title("differrent optimiszer's val_loss")
plt.legend()
plt.savefig("optimizer's loss.png")
plt.show()