import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

#setando os dados do csv instalado no kaggle na variavel data
data = pd.read_csv('titanic.csv')

#A ideia aqui é ver e saber os tipos de dados das colunas, porque o algoritimo de ml só entende números
#o print mostra a quantidade de dados nulos em cada coluna
data.info()
print(data.isnull().sum())


#dataCleaning 
def preprocess_data(df):
  #tira as colunas que não importam para o algoritimos, como o nome por exemplo
  df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

  #preenche os dados nulos da coluna Embarker com o valor S que seria no caso desse dataset um dos portos de embarque
  df["Embarked"].fillna("S", inplace=True)
  df.drop(columns = ["Embarked"],inplace=True )

  #coneverte o gênero para número
  df["Sex"] = df["Sex"].map({'male':1,'female':0})

  #Featur engeneering
  df["FamilySize"] = df["SibSp"] + df["Parch"]
  #se FamilySize = 0 então IsAlone = 1, se não IsAlone = 0
  df["IsAlone"] = np.where(df["FamilySize"] == 0 ,1,0)

  df["FareBin"] = pd.qcut(df["Fare"],4,labels=False)
  df["AgeBin"] = pd.qcut(df["Age"],bins=[0,12,20,40,60, np.inf],labels=False)

  return df

def fill_missing_ages(df):
  print("Parei aqui")
