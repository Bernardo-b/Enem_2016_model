import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the train data
train = 'testfiles/train.csv'
dataset = pd.read_csv(train)
# filter dataset to remove null values in the important columns and organizes
dataset = dataset[dataset['TX_RESPOSTAS_MT'].notna()]
dataset = dataset[dataset['NU_NOTA_MT'].notna()]
dataset.index = list(range(len(dataset)))

# name the y with the score columns
y = dataset['NU_NOTA_MT']

# transforms the answers vector to unique answers in each question
# instead of ABDECB it is A1 B2 D3 E4 C5 B5
# and puts it in string so to be transformed in a binary vector
c = []
for i in range(len(dataset.TX_RESPOSTAS_MT)):
    b = list(dataset.TX_RESPOSTAS_MT[i])
    for a in range(len(b)):
        b[a] = b[a] + str(a+1)
    c.append(b)
# transfors the array for the CountVectorizer
separador = ' '
for i in range(len(c)):
    c[i] = separador.join(c[i])

# creating the features names for the vector
vetores = []
abc = ['A', 'B', 'C', 'D', 'E']
for i in range(45):
    for j in abc:
        vetores.append(j+str(i+1))
# creates the vector
vec = CountVectorizer(lowercase=False, analyzer='char_wb', max_features=5*45,
                      vocabulary=vetores)
# transforms the vector in array
X = vec.fit_transform(c)

model = LinearRegression()

model.fit(X, y)


# X = pd.DataFrame(c, columns=[i for i in range(1, 46)])

# model = LinearRegression(fit_intercept=False)
#
# model.fit(X, y)
