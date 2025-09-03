# MachineLearning

Nesse repositório contém alguns projetos em que ajudam a entender melhor o funcionamento do aprendizado de maquina junto com uma breve descrição de cada modelo/projeto, erro encontrado, e uma sobre alguns métodos importantes que possam me ajudar em problemas futuros.

-Previsão de sobrevicência no titanic - Modelo K-Near Neighbors (KNN)
    Esse é um projeto base para eu entender o básico sobre as bibliotecas do matlibplot, scikit-learn e seaborn.
    A ideia desse projeto é ter um modelo de aprendizado de maquina que leia um .csv que contém dados de passageiros hipotéticos do titanic e se sobreviveram ou não a ele.
    -   Para iniciar esse projeto eu comecei com uma limpeza dos dados, inserindo valores a campos nulos e limpando as colunas desnecessárias.
    -   Eu separo então os dados em features e targets em que vão servir de maneira análoga à Flash cards em que X é a parte da frente de um flash card e Y seria a parte traseira, que iria conter a resposta
    
    Essa parte eu acredito que seria bem importante deixar salvo aqui principalente se eu for usar novamente esse modelo do KNN:

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
         
    Esse código transforma os valores que tem na base de dados em números entre 0 e 1. Isso serve para fazer com que a "IA" consiga fazer toda a correlação dos dados conseguindo "Comparar" eles.
    Um exemplo para entender isso, seria o caso de Age e Fare, se você for calcular a "distância" entre dois passageiros (como o KNN faz), a Tarifa, por ter valores muito maiores, vai dominar completamente o cálculo. A Idade, com seus valores menores, quase não fará diferença. É como tentar comparar a altura de uma pessoa (em metros) com o peso de um elefante (em quilos) diretamente.

    -   Depois disso eu crio um dicionário de parâmetros para testar os modelos e escolho o com melhor acurácia
    -   Depois faço o modelo fazer as previsões
    -   Por fim crio um gráfico com matlibplot para mostrar a confusion matrix

    O maior problema que encontrei durante esse projeto, foi que o modelo estava com 100% de acuracia. Isso me deixou desconfiado e depois de um tempo validado, eu descobri que o problema era na verdade no dataset que eu estava usando. Nele todos os Homens morriam, e todas as mulheres sobreviviam, então o que o modelo estava fazendo era considerar o gênero da pessoa e dizer que morria caso homem e sobrevivia caso mulher

-Previsão do tempo - RandomForestClassifier
    Para esse projeto, a ideia inicial era criar um modelo para prever o clima.
    Para desenvolver o projeto eu peguei uma basse de dados do histórico de temperatura em Seattle, fiz algumas limpeza nos dados, convertendo tudo para inteiros, e removendo colunas desnecessárias.
    Para testes foi treinado o modelo utilizando as features de data, precipitação, temperatura maxima e minima e velocidade do vento, para prever o clima


  
