import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score





raw_dataframe = pd.read_csv('aoe_data.csv')


# Removal of unnecessary columns
data_frame_temp = raw_dataframe.drop([raw_dataframe.columns[0], "winner", "map", "duration", "match_id", "dataset", "p1_xpos", "p2_xpos", "p1_ypos", "p2_ypos", "difficulty", "map_size"], axis=1)


# Seperating out the players civ choice
civ_temp1 = data_frame_temp["p1_civ"]
civ_temp2 = data_frame_temp["p2_civ"]


"""
Naive Bayes Section
"""
# Making Data work for Naive Bayes
nb_civ1 = pd.get_dummies(civ_temp1, columns=["p1_civ"], prefix="civ1", dtype=int)
nb_civ2 = pd.get_dummies(civ_temp2, columns=["p2_civ"], prefix="civ2" , dtype=int)
nb_data_frame = pd.concat([nb_civ1, nb_civ2], axis=1)


# Spliting into X and y
X = nb_data_frame
y = raw_dataframe["winner"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model with Naive Bayes
model = BernoulliNB()
model.fit(x_train, y_train)


# Results of Naive Bayes
print(accuracy_score(y_test, model.predict(x_test)))


"""
Neural Network Section
"""
# Seperating out the civs into distinct columns for both players
civ1 = pd.get_dummies(civ_temp1, columns=["p1_civ"], prefix="civ", dtype=int)
civ2 = pd.get_dummies(civ_temp2, columns=["p2_civ"], prefix="civ" , dtype=int)

# Process for making civ columns joined 
# player one being positive and player 2 being negative 
# (mirror matches are 0 in all civ columns)
civ_joined = data_frame_temp.drop(["p1_civ", "p2_civ", "elo"], axis=1)
for column in civ1:
    civ_joined[column] = civ1[column] - civ2[column]


# Normalizing Elo Value
elo_temp = data_frame_temp["elo"]
mean = elo_temp.mean()
std = elo_temp.std()
elo = (elo_temp - mean) / std


# Making X and Y
# X = pd.concat([civ_joined, elo, raw_dataframe["duration"]], axis=1)
# Commented out to make the model more accurate by removing duration and elo

X = civ_joined
y = raw_dataframe["winner"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model with Neural Network
NN_model = MLPClassifier(max_iter=1000, activation="logistic", hidden_layer_sizes=(32))
NN_model.fit(x_train, y_train)


print(accuracy_score(y_test, NN_model.predict(x_test))) # Results of Neural Network model.

