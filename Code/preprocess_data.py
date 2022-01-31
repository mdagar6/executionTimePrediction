import pickle
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder

random.seed(9845)
np.random.seed(9845)

architectures = ['GTX1650', 'K20', 'K80', 'M60', 'P100', 'T4', 'TitanXp', 'V100']
frames_X_train = []
frames_X_test = []
frames_y_train = []
frames_y_test = []
for arch in architectures:    
    data = pandas.read_pickle(f"./../data/pkl/{arch}_median.pkl")

    label = data.pop("time")
    data.pop("bench")
    data.pop("app")
    data.pop("dataset")
    data.pop("name")
    
    data['arch'] = [arch] * len(label)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=5642)
    frames_X_train.append(X_train)
    frames_X_test.append(X_test)
    frames_y_train.append(y_train)
    frames_y_test.append(y_test)

X_train = pandas.concat(frames_X_train)
X_test = pandas.concat(frames_X_test)
y_train = pandas.concat(frames_y_train)
y_test = pandas.concat(frames_y_test)

X_train = pandas.concat((X_train,pandas.get_dummies(X_train.arch)),1)
X_test = pandas.concat((X_test,pandas.get_dummies(X_test.arch)),1)
X_train.pop("arch")
X_test.pop("arch")

X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())
  
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
y_train, y_test = np.log(y_train.astype(float)), np.log(y_test.astype(float))
    
train_dataset = {"X_train": X_train, "y_train": y_train}
test_dataset = {"X_test": X_test, "y_test": y_test}
    
with open(f"./../data/train/train.pkl", 'wb') as file:
    pickle.dump(train_dataset, file)

with open(f"./../data/test/test.pkl", 'wb') as file:
    pickle.dump(test_dataset, file)