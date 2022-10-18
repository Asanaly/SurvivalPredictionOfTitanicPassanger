import numpy as np
import csv
import tensorflow as tf
from keras import models
from keras import layers
from keras import regularizers

import pandas as pd

data = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
real_test = pd.read_csv("/content/test.csv")
gen_sub = pd.read_csv("/content/gender_submission.csv")

#Data cleaning

def clean_data(data):
  data = data.drop( ["PassengerId", "Name", "Ticket", "Cabin"] , axis = 1)

  colls = ["Pclass", "Age" , "SibSp" , "Parch"]
  for col in colls:
    data[col].fillna(data[col].median(), inplace = True)
  
  data.Embarked.fillna("U", inplace = True)
  return data

data = clean_data(data)
test = clean_data(test)

def to_cat(data):

  used = []
  newD = data

  for i,val in enumerate(data):
    if not val in used:
      used.append(val)
    newD[i] = used.index(val)
  return newD

data["Sex"] = to_cat(data["Sex"])
test["Embarked"] = to_cat(test["Embarked"])
test["Sex"] = to_cat(test["Sex"])
data["Embarked"] = to_cat(data["Embarked"])

AGEMAX = 80
FAREMAX = 514
PCLASSMAX = 3 

data["Age"] /= AGEMAX 
test["Age"] /= AGEMAX
data["Fare"] /= AGEMAX 
test["Fare"] /= AGEMAX
data["Pclass"] /= PCLASSMAX 
test["Pclass"] /= PCLASSMAX

data_labels = data["Survived"]
test_labels = data["Survived"]
data = data.drop( ["Survived"] , axis = 1)

data = data.to_numpy()
test = test.to_numpy()

val_data = data[600:891]
data = data[:600]

data_labels = data_labels.to_numpy()
test_labels = test_labels.to_numpy()

val_labels = data_labels[600:]
data_labels = data_labels[:600]

#Model Compilation

model = models.Sequential()

model.add(layers.Dense(32, kernel_regularizer = regularizers.l1_l2(0.001), activation = 'relu' ) )
model.add(layers.Dropout(0.1))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l1_l2(0.001), activation = 'relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(4, kernel_regularizer = regularizers.l1_l2(0.001), activation = 'relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

data = data.astype(float)
test = test.astype(float)
data_labels = data_labels.astype(float)
val_data = val_data.astype(float)
val_labels = val_labels.astype(float)

print(test)
history = model.fit(data, data_labels, epochs = 35, batch_size = 256, validation_data = (val_data, val_labels))

ans = model.predict(test)

#Data vizualization

import matplotlib.pyplot as plt

finalAnsId = {}

for i,val in enumerate(ans):
  print(real_test["PassengerId"][i], val)
  finalAnsId[real_test["PassengerId"][i]] =  val

loss = history.history['accuracy']
print(history.history.keys())
val_loss = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training_loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

#Data output as csv file

import math

for i,val in enumerate(gen_sub["PassengerId"]):
  if math.isnan(finalAnsId[val][0]):
    gen_sub["Survived"][i] = 1
  else:
   # print(int(round(finalAnsId[val][0])))
    gen_sub["Survived"][i] = int(round(finalAnsId[val][0]))

gen_sub["Survived"] = gen_sub["Survived"].astype(int)
print(gen_sub.drop)

DF = pd.DataFrame(gen_sub)

# save the dataframe as a csv file
DF.to_csv("/content/data_ans.csv")
