import pandas
import pickle
import catboost as cb
import numpy as np

#Read Data
data = pandas.read_pickle(f"../data/train/train.pkl")
X = data["X_train"]
y = data["y_train"]

##Train CatBoost
print("Training CatBoost")
train_dataset = cb.Pool(X, y)
model2 = cb.CatBoostRegressor(depth = 16, iterations = 200)
model2.fit(train_dataset)
print("Save Model")
filename = f"../trained_model/catboost.pkl"
pickle.dump(model2, open(filename, 'wb'))
