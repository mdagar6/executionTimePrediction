import pandas
import pickle
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

#Read Data
data = pandas.read_pickle(f"../data/train/train.pkl")
X = data["X_train"]
y = data["y_train"]
data = pandas.read_pickle(f"../data/test/test.pkl")
X_test = data["X_test"]
y_test = data["y_test"]

# Train TabNetRegressor
print("Training TabNet")
y_test = np.reshape(y_test, (-1, 1))
y = np.reshape(y, (-1, 1))
model3 = TabNetRegressor(optimizer_params = dict(lr=1e-3), seed = 0)  
model3.fit(
  X, y,
  eval_set=[(X_test, y_test)]
)
print("Save Model")
filename = f"../trained_model/tabnet.pkl"
pickle.dump(model3, open(filename, 'wb'))
