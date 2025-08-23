# MachineLearning

Nesse repositório contém alguns projetos em que ajudam a entender melhor o funcionamento do aprendizado de maqui e deixo aqui uma breve descrição de cada projeto, erro encontrado, e uma descrição sobre alguns métodos importantes que possam me ajudar em problemas futuros.

-Previsão de sobrevicência no titanic - Modelo K-Near Neighbors (KNN)
    Esse é um projeto base para eu entender o básico sobre as bibliotecas do matlibplot, scikit-learn e seaborn.
    A ideia desse projeto é ter um modelo de aprendizado de maquina que leia um .csv que contém dados de passageiros hipotéticos do titanic e se sobreviveram ou não a ele.
    -   Para iniciar esse projeto eu comecei com uma limpeza dos dados, inserindo valores a campos nulos e limpando as colunas desnecessárias.
    -   Eu separo então os dados em features e targets em que vão servir de maneira análoga à Flash cards em que X é a parte da frente de um flash card e Y seria a parte traseira, que iria conter a resposta
    
    Essa parte eu acredito que seria bem importante deixar salvo aqui principalente se eu for usar novamente esse modelo do KNN:

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
         
    Esse código transforma os valores que tem na base de dados em números entre 0 e 1. Isso serve para fazer com que a IA consiga fazer toda a correlação dos dados conseguindo "Comparar" eles.
    Um exemplo para entender isso, seria o caso de Age e Fare, se você for calcular a "distância" entre dois passageiros (como o KNN faz), a Tarifa, por ter valores muito maiores, vai dominar completamente o cálculo. A Idade, com seus valores menores, quase não fará diferença. É como tentar comparar a altura de uma pessoa (em metros) com o peso de um elefante (em quilos) diretamente.

    -   Depois disso eu crio um dicionário de parâmetros para testar os modelos e escolho o com melhor acurácia
    -   Depois faço o modelo fazer as previsões
    -   Por fim crio um gráfico com matçibplot para mostrar a confusion matrix
    
    