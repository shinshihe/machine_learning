import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder


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
# X = data
le = StandardScaler().fit(data)
X = le.transform(data)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,shuffle = True)



clf =  KNeighborsClassifier(n_neighbors=3,weights='distance')
clf.fit(x_train, y_train)
y_pre = clf.predict(x_test)

acc = accuracy_score(y_test,y_pre)
print("The accuracy is",acc)