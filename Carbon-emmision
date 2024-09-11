import pandas as pd
a=pd.read_csv("Agrofood_co2_emission.csv")
a.isna()
from sklearn import preprocessing
y=preprocessing.LabelEncoder()
a["Area"]=y.fit_transform(a["Area"])
a["Year"]=y.fit_transform(a["Year"])
a["Savanna fires"]=y.fit_transform(a["Savanna fires"])
a["Forest fires"]=y.fit_transform(a["Forest fires"])
a["Crop Residues"]=y.fit_transform(a["Crop Residues"])
a["Rice Cultivation"]=y.fit_transform(a["Rice Cultivation"])
a["Forestland"]=y.fit_transform(a["Forestland"])
a["Net Forest conversion"]=y.fit_transform(a["Net Forest conversion"])
a["IPPU"]=y.fit_transform(a["IPPU"])
a["IPPU"].unique()
a["Net Forest conversion"].unique()
a["Forestland"].unique()
a["Rice Cultivation"].unique()
a["Crop Residues"].unique()
a["Forest fires"].unique()
a["Savanna fires"].unique()
a["Year"].unique()
a["Area"].unique()
print("******GRAPHICAL MODEL OF THE GIVEN DATA ******")
import seaborn as sd
sd.boxplot(x="Area",y="Year",data=a)
sd.boxplot(x="Savanna fires",y="Forest fires",data=a)
sd.boxplot(x="Crop Residues",y="Rice Cultivation",data=a)
sd.boxplot(x="Forestland",y="IPPU",data=a)
print("******DISTRIBUSION PLOT******")
sd.distplot(a.Area)
sd.distplot(a.Year)
sd.distplot(a.Forestland)
sd.distplot(a.IPPU)
print("******BARPLOT******")
sd.barplot(x="Area",y="Year",data=a)
sd.barplot(x="Savanna fires",y="Forest fires",data=a)
sd.barplot(x="Crop Residues",y="Rice Cultivation",data=a)
sd.barplot(x="Forestland",y="IPPU",data=a)
print("******LMPLOT******")
sd.lmplot(x="Area",y="Year",data=a)
sd.lmplot(x="Savanna fires",y="Forest fires",data=a)
sd.lmplot(x="Crop Residues",y="Rice Cultivation",data=a)
sd.lmplot(x="Forestland",y="IPPU",data=a)
from matplotlib import pyplot as plt
plt.hist(a["Area"])
plt.hist(a["Year"])
plt.hist(a["Savanna fires"])
plt.hist(a["Forest fires"])
plt.hist(a["Crop Residues"])
plt.hist(a["Rice Cultivation"])
plt.hist(a["Forestland"])
plt.hist(a["IPPU"])
print("*******SCATTER PLOT*******")
from matplotlib import pyplot as plt
plt.scatter(a["IPPU"],a["On-farm energy use"])
print("******* PLOT GRAPH *******")
from matplotlib import pyplot as plt
plt.plot(a["Manure left on Pasture"],a["Manure applied to Soils"])
print("******SPLIT TEST AND TRAIN DATA******")
x=a.iloc[:,:-1].values
print(x)
y=a["Area"]
print(y)
print("******TRAIN_TEST_SPLIT ALGORITHM******")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.20,shuffle=True)
print(x_train)
print(x_test)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
S = LogisticRegression()
S.fit(x_train_imputed, y_train)
predictions = S.predict(x_test_imputed)
print("Accuracy score:", accuracy_score(y_test, predictions))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
print("******PRINTING THE TRAIN DATAS******")
print(x_train)
print(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
S = RandomForestClassifier()
S.fit(x_train_imputed, y_train)
predictions = S.predict(x_test_imputed)
print("Accuracy score:", accuracy_score(y_test, predictions))
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
S = GaussianNB()
S.fit(x_train_imputed, y_train)
predictions = S.predict(x_test_imputed)
print("Accuracy score:", accuracy_score(y_test, predictions))
from matplotlib import pyplot as plt
import numpy as np
cars = ['Area', 'Savanna fires', 'Food Household Consumption',
        'Food Retail', 'Food Processing', 'total_emission']
data = [23, 17, 35, 29, 12, 41]
fig = plt.figure(figsize =(10, 7))
plt.pie(data)
plt.show()
