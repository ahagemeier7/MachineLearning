import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error,confusion_matrix

#Carregando e validando dados
data = pd.read_csv("seattle-weather.csv")

def data_preprocessing(df):
    #Converte a string date (20-01-2025) para colunas separadas, contendo somente números para o modelo entender
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.day_of_week
    df["dayofyear"] = df["date"].dt.day_of_year
    
    df["weather"] = df["weather"].map({'drizzle': 0, 'rain': 1, 'sun': 2, 'snow': 3, 'fog':4})
    
    return df
    
data = data_preprocessing(data)

data = data.drop(columns=["date"])
# definindo a data como feature e cada valor a ser previsto pelo Modelo como os targets
x = data.drop(columns=["weather"])
y = data["weather"]

#definindo as variaveis de target e feature de treino e teste
x_train_wea, x_test_wea,y_train_wea,y_test_wea = train_test_split(x,y,test_size=0.25,random_state=42)

rf_weather = RandomForestClassifier(n_estimators=100, random_state=42)
rf_weather.fit(x_train_wea, y_train_wea)


#fazendo as previsões
y_pred_wea = rf_weather.predict(x_test_wea)
accurracy_weather = accuracy_score(y_test_wea,y_pred_wea,normalize=True)
matrix = confusion_matrix(y_test_wea,y_pred_wea)
print("-------------weather-------------")
print(accurracy_weather)
print(matrix)
