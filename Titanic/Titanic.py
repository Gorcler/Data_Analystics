import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing, tree, model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv("Titanic_dataset.csv")
fig = plt.figure(figsize=(18,6))

#1)--------------visual analytics---------------
plt.subplot2grid((2,3), (0,0))
data.Survived.value_counts(normalize=True).plot(kind="barh")
plt.title("Survived")
plt.ylabel("Survived")

plt.subplot2grid((2,3), (0,1))
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age_Survived")
plt.ylabel("Age")

plt.subplot2grid((2,3), (0,2))
data.Pclass.value_counts(normalize=True).plot(kind="barh")
plt.title("Class_Structure")
plt.ylabel("Class")

plt.subplot2grid((2,3), (1,0), colspan=2)
for x in [1,2,3]:
    data.Age[data.Pclass == x].plot(kind="kde")
plt.title("Class_Age")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((2,3), (1,2))
data.Embarked.value_counts(normalize=True).plot(kind="barh")
plt.title("Embarked")
plt.ylabel("City")

plt.show()

#Broader_visualization
fig = plt.figure(figsize=(18, 6))
female_color = "#FA0000"

plt.subplot2grid((3,4), (0,0))
data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("Survived")

plt.subplot2grid((3,4), (0,1))
data.Survived[data.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("Men_Survived")

plt.subplot2grid((3,4), (0,2))
data.Survived[data.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.7, color = female_color)
plt.title("Female_Survived")

plt.subplot2grid((3,4), (0,3))
data.Sex[data.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.7, color = [female_color, 'b'])
plt.title("Sex_Survived")

plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1,2,3]:
    data.Survived[data.Pclass == x].plot(kind="kde")
plt.title("Class_Survived")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((3,4), (2,0))
data.Survived[(data.Sex == "male") & (data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("1ste Class male_Survival")

plt.subplot2grid((3,4), (2,1))
data.Survived[(data.Sex == "male") & (data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("3rd Class male_Survival")

plt.subplot2grid((3,4), (2,2))
data.Survived[(data.Sex == "female") & (data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.7, color = female_color)
plt.title("1ste Class female_Survival")

plt.subplot2grid((3,4), (2,3))
data.Survived[(data.Sex == "female") & (data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.7, color = female_color)
plt.title("3rd Class female_Survival")

plt.show()

#Prediction Algoritme
def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

clean_data(data)

model_1 = linear_model.LinearRegression()
model_2 = RandomForestRegressor()

target = data["Survived"].values
features = data[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

X_train , X_test , Y_train, Y_test = train_test_split(features , target, test_size=0.2)

print(X_train)

classifier_1 = model_1.fit(X_train, Y_train)
print(classifier_1.score(X_train, Y_train)*100, "% Accuracy")

classifier_2 = model_2.fit(X_train, Y_train)
print(classifier_2.score(X_train, Y_train)*100, "% Accuracy")

predict_1 = model_1.predict(X_test)
predict_2 = model_2.predict(X_test)

#Model_acurracy_Visualization
fig = plt.figure(figsize=(18, 6))

plt.subplot2grid((2,2), (0,0), colspan=2)
plt.scatter(predict_1, Y_test, alpha=0.35)
plt.ylabel("true result")
plt.xlabel("prediction")
plt.title("Prediction_Linear")

plt.subplot2grid((2,2), (1,0), colspan=2)
plt.scatter(predict_2, Y_test, alpha=0.35)
plt.ylabel("true result")
plt.xlabel("prediction")
plt.title("Prediction_RandomForestRegressor")

plt.show()
