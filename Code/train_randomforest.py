import pandas
import pickle
from sklearn.ensemble import RandomForestRegressor
#Read Data
data = pandas.read_pickle(f"../data/train/train.pkl")
X = data["X_train"]
y = data["y_train"]

# Train RandomForest
print("Training Random Forest")
model1 = RandomForestRegressor(max_depth=16, random_state=9756)
model1.fit(X, y)
print("Save Model")
filename = f"../trained_model/randomforest.pkl"
pickle.dump(model1, open(filename, 'wb'))

