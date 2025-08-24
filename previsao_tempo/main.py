import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error,confusion_matrix

#Carregando e validando dados
data = pd.read_csv("previsao_tempo/seattle-weather.csv")

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
x = data.drop(columns=["precipitation","temp_max","temp_min","wind","weather"])

y_precipitation = data["precipitation"]
y_temp_max = data["temp_max"]
y_temp_min = data["temp_min"]
y_wind = data["wind"]
y_weather = data["weather"]

#definindo as variaveis de target e feature de treino e teste
x_train_prec, x_test_prec,y_train_prec,y_test_prec = train_test_split(x,y_precipitation,test_size=0.20,random_state=42)
x_train_tempmax, x_test_tempmax,y_train_tempmax,y_test_tempmax = train_test_split(x,y_temp_max,test_size=0.20,random_state=42)
x_train_tempmin, x_test_tempmin,y_train_tempmin,y_test_tempmin = train_test_split(x,y_temp_min,test_size=0.20,random_state=42)
x_train_wind, x_test_wind,y_train_wind,y_test_wind = train_test_split(x,y_wind,test_size=0.20,random_state=42)
x_train_wea, x_test_wea,y_train_wea,y_test_wea = train_test_split(x,y_weather,test_size=0.20,random_state=42)

#instanciando os modelos
lr_precipitation = LinearRegression()
lr_precipitation.fit(x_train_prec, y_train_prec)

lr_tempmax = LinearRegression()
lr_tempmax.fit(x_train_tempmax,y_train_prec)

lr_tempmin = LinearRegression()
lr_tempmin.fit(x_train_tempmin,y_train_tempmin)

lr_wind = LinearRegression()
lr_wind.fit(x_train_wind,y_train_wind)

rf_weather = RandomForestClassifier(n_estimators=100, random_state=42)
rf_weather.fit(x_train_wea, y_train_wea)


#fazendo as previsões
y_pred_prec = lr_precipitation.predict(x_test_prec)
rmse_prec = np.sqrt(mean_squared_error(y_test_prec, y_pred_prec))

y_pred_tempmax = lr_tempmax.predict(x_test_tempmax)
rmse_tempmax = np.sqrt(mean_squared_error(y_test_tempmax, y_pred_tempmax))

y_pred_tempmin = lr_tempmin.predict(x_test_tempmin)
rmse_tempmin = np.sqrt(mean_squared_error(y_test_tempmin, y_pred_tempmin))

y_pred_wind = lr_wind.predict(x_test_wind)
rmse_wind = np.sqrt(mean_squared_error(y_test_wind, y_pred_wind))

y_pred_wea = rf_weather.predict(x_test_wea)
accurracy_weather = accuracy_score(y_test_wea,y_pred_wea,normalize=False)
matrix = confusion_matrix(y_test_wea,y_pred_wea)

print("-------------precepitation-------------")
print(rmse_prec)
print("----------------temp_max---------------")
print(rmse_tempmax)
print("----------------temp_min---------------")
print(rmse_tempmin)
print("-----------------wind------------------")
print(rmse_wind)
print("-------------weather-------------")
print(accurracy_weather)
print(matrix)
