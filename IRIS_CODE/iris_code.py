import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

iris_data = pd.read_csv("IRIS.csv")
specie_name = ["Iris_setosa", "Iris_versocolor", "Iris_virginca"]
value = []

#---visual Data Analysis---

sns.set_style("whitegrid")
sns.pairplot(iris_data,hue="species",height=3);
plt.show()

#-------Data Management----------
def clean_data(data):
    data.loc[data["species"] == "Iris-setosa", "species"] = 0
    data.loc[data["species"] == "Iris-versicolor", "species"] = 1
    data.loc[data["species"] == "Iris-virginica", "species"] = 2

clean_data(iris_data)

features = np.array(iris_data.drop(["species"], 1))
target = np.array(iris_data["species"])

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = 9)

y_train = y_train.astype("int")
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    classifier_1 = knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    y_predict = y_predict.astype("int")
    scores.append(classifier_1.score(x_train, y_train))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

model_2 = linear_model.LinearRegression()
classifier_1 = model_2.fit(x_train, y_train)
print("Linear_Model: ", round((classifier_1.score(x_train, y_train)*100), 3), "% Accuracy")

prediction = model_2.predict(x_test)
prediction = prediction.astype("int")
predictions = []
result = " "
y_test = y_test.astype("int")
print("")

for x in range(0, len(y_test)):
    predictions.append(round(prediction[x]))

    if y_test[x] == predictions[x]:
        result = "Correct"
    else:
        result = "Incorrect"

    print("Predictions/True_value:\n",specie_name[predictions[x]], " / ",specie_name[y_test[x]],"\nResult: ", result,"\n")

