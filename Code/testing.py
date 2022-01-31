import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np

#Read Data
data = pandas.read_pickle(f"../data/test/test.pkl")
X_test = data["X_test"]
y_test = data["y_test"]

models = {}
print("Load Models")
# RandomForest
filename = f"../trained_model/randomforest.pkl"
model1 = pickle.load( open(filename, 'rb'))
models["RandomForest"] = model1

# CatBoost
filename = f"../trained_model/catboost.pkl"
model2 = pickle.load(open(filename, 'rb'))
models["catboost"] = model2

# TabNetRegressor
filename = f"../trained_model/tabnet.pkl"
model3 = pickle.load( open(filename, 'rb'))
models["tabnet"] = model3

def MAPE(y_pred, y_true):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for arch in models:
	model = models[arch]
	
	y_pred = model.predict(X_test)

	res = []
	for i in range(len(y_pred)):
	    m = MAPE(y_pred[i], y_test[i])
	    res.append(m)
	    
	bins=[0, 10, 25, 50, 100, 200, float("Inf")]
	count, edges = np.histogram(res, bins=bins)

	x = np.arange(len(count))
	xticklabels = []
	for i in range(0,len(edges)-1):
	    xticklabels.append(str(edges[i]).rstrip('0').rstrip('.')+" - "+str(edges[i+1]).rstrip('0').rstrip('.')+"%")

	plt.bar(x, count, tick_label=xticklabels, color = 'blue')
	plt.ylabel('Count')
	plt.xlabel('MAPE')
	plt.tick_params(axis='x',rotation=60)
	plt.title(f"Distribution of prediction errors - {arch}")
	plt.tight_layout()
	plt.savefig(f"{arch}.pdf")
	plt.show()
	print(f"{arch} Count: ", count)

