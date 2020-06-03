import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import csv


# load the training and testing data
train = 'testfiles/train.csv'
X = pd.read_csv(train)
test = 'testfiles/test.csv'
datatest = pd.read_csv(test)

# deletes any rows which the score is null
X = X[X['NU_NOTA_MT'].notna()]

# scores answer
y = X['NU_NOTA_MT']

#gets only the columns which the values are numerical
corretas = []
for i in range(len(datatest.iloc[0])):
    if type(datatest.iloc[8,i]) is not str:
        corretas.append(i)
X = X[datatest.columns[corretas]]

numero_de_ins = datatest.NU_INSCRICAO

#matches the collumns
datatest = datatest[datatest.columns[corretas]]

# replaces any null values with the mean of the column
imp = SimpleImputer()
imp.fit(X)
X = imp.transform(X)
imp.fit(datatest)
datatest = imp.transform(datatest)

# linear regression
lin = LinearRegression()
lin.fit(X, y)

# predicts the result
t = lin.predict(datatest)
pd.DataFrame()
