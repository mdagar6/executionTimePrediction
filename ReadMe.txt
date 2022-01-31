There are three folders:
Code
data

Dependencies:
module load python-3.7
pip3 install pandas
pip3 install -U scikit-learn
pip3 install pytorch-tabnet
pip3 install catboost
pip3 install pandas
pip3 install matplotlib


Code:
preprocess_data.py -> This code pre-processed the data and split it into train and test data.
We already preprocessed the data, which is available in the data/test and data/train folder. Please make a note that if you run this code again, it will create a different split for train and test, which may not produce the same numbers as reported in our report. 
To run this use the below command:
python3 preprocess_data.py 

train_tabnet.py -> 
Train Tabnet Model.
To run this use the below command:
python3 train_tabnet.py 

train_catboost.py -> 
Train CatBoost Model. 
To run this use the below command:
python3 train_catboost.py

train_randomforest.py -> 
Train Randomforest Model.
To run this use the below command:
python3 train_randomforest.py

testing.py -> 
Test all three models and generate the graphs. 
To run this use the below command:
python3 testing.py

NOTE: YOU DON'T NEED TO PROVIDE A PATH TO TRAIN OR TEST DATA OR TRAINED MODEL WHILE RUNNING THE ABOVE FILES BECAUSE IT WILL USE THE DEFAULT STRUCTURE OF ALL THE FOLDERS. SO PLEASE DON'T MAKE ANY CHANGES TO THE FOLDER STRUCTURES. 

Data:
pkl folder contains unprocessed data for all the eight GPU architectures. 
train folder contains train pre-processed data.
test folder contains test pre-processed data.

trained_model:
After training, models will be saved in this folder. This folder contains models trained by us.
Please make a note that after training reproducibility is not 100% guaranteed even after using the same seed value.