import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

#setando os dados do csv instalado no kaggle na variavel data
data = pd.read_csv('titanic.csv')

# Imprime o número de linhas originais
print(f"Número de linhas originais: {len(data)}")

# Conta as duplicatas antes de remover
num_duplicatas_antes = data.duplicated().sum()
print(f"Número de linhas duplicadas (antes de remover): {num_duplicatas_antes}")

#A ideia aqui é ver e saber os tipos de dados das colunas, porque o algoritimo de ml só entende números
#o print mostra a quantidade de dados nulos em cada coluna
data.info()
print(data.isnull().sum())

print(data.columns)

#dataCleaning 
def preprocess_data(df):
  #tira as colunas que não importam para o algoritimos, como o nome por exemplo
  df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

  #preenche os dados nulos da coluna Embarker com o valor S que seria no caso desse dataset um dos portos de embarque
  df["Embarked"].fillna("S", inplace=True)
  df.drop(columns = ["Embarked"],inplace=True )

  fill_missing_ages(df)

  #coneverte o gênero para número
  df["Sex"] = df["Sex"].map({'male':1,'female':0})

  #Featur engeneering
  df["FamilySize"] = df["SibSp"] + df["Parch"]
  #se FamilySize = 0 então IsAlone = 1, se não IsAlone = 0
  df["IsAlone"] = np.where(df["FamilySize"] == 0 ,1,0)

  df["FareBin"] = pd.qcut(df["Fare"],4,labels=False)
  df["AgeBin"] = pd.cut(df["Age"],bins=[0,12,20,40,60, np.inf],labels=False)

  return df

def fill_missing_ages(df):
  #preenche os dados nulos da coluna Age com a mediana de idade de cada classe
  age_fill_map = {}
  for pclass in df["Pclass"].unique():
    if pclass not in age_fill_map:
      age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

  #Seta o valor da idade com a mediana de cada classe
  #Ou seja para cada linha ele valida se a Idade é nula e preenche com a mediana da classe do passageiro
  df["Age"] = df.apply(lambda row:age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],axis=1)


data = preprocess_data(data)

# Remove as duplicatas após o pré-processamento, pois novas duplicatas podem ter sido criadas.
data.drop_duplicates(inplace=True)
print(f"Número de linhas após remover duplicatas (após pré-processamento): {len(data)}")

print("\nMissing values after preprocessing:")
print(data.isnull().sum())

print("\nCorrelation Matrix after preprocessing:")
print(data.corr(numeric_only=True)["Survived"].sort_values(ascending=False))

#separa os dados em x e y, onde x são as features e y o target
x = data.drop(columns=["Survived", "Sex"]) # Removendo 'Sex' para evitar data leakage devido à correlação perfeita
y = data["Survived"] #Aqui tem a resposta, sendo o que o modelo precisa chutar

#criando os dados de treino e teste em que 25% dos dados serão de teste e 75% de treino
#essas 4 variaveis são por exemplo um flashcard, em que x_train é a frente do flashcard e y_train o verso
#Então enquanto estiver trainando o modelo ele vai ver a frente e o verso do flashcard
#mas enquanto estiver testando o modelo ele só vai ver a frente do flashcard e tentar adivinhar o verso
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# ML preprocessing
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

#normaliza os dados para os valores das colunas terem somente um intervalo entra 1 e 0 para funcionar o as comparações entre diferentes "unidades" no modelo do KNN
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#definindo parametros para o modelo, o modelo é o KNN
#ou seja ele vai perguntar para os vizinhos mais próximos qual a resposta correta
def tune_model(x_train,y_train):
  param_grid = {
    "n_neighbors": range(1,21),
    "metric": ["euclidean","manhattan","minkowski"],
    "weights": ["uniform","distance"]
  }

  model = KNeighborsClassifier()
  grid_search = GridSearchCV(model,param_grid,cv=5,n_jobs=-1,scoring="accuracy")
  
  #fit é o treino do modelo
  grid_search.fit(x_train,y_train)
  return grid_search.best_estimator_

best_model = tune_model(x_train,y_train)


#fazendo as previsões
def evaluate_model(model,x_test,y_test):
  prediction = model.predict(x_test)
  accuracy = accuracy_score(y_test,prediction)
  matrix = confusion_matrix(y_test,prediction)
  return accuracy,matrix

accuracy,matrix = evaluate_model(best_model,x_test,y_test)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)


def plot_model(matrix):
  plt.figure(figsize=(10,7))
  sns.heatmap(matrix, annot=True, fmt="d",xticklabels=["Survived","Not Survived"], yticklabels=["Not Survived", "Survived"])
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.show()

plot_model(matrix)
