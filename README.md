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
    -   Por fim crio um gráfico com matlibplot para mostrar a confusion matrix

    O maior problema que encontrei durante esse projeto, foi que o modelo estava com 100% de acuracia. Isso me deixou desconfiado e depois de um tempo validado, eu descobri que o problema era na verdade no dataset que eu estava usando. Nele todos os Homens morriam, e todas as mulheres sobreviviam, então o que o modelo estava fazendo era considerar o gênero da pessoa e dizer que morria caso homem e sobrevivia caso mulher

-Previsão do tempo - LinearRegression e RandomForestClassifier
    Para esse projeto, a ideia inicial era criar um modelo para prever a temperatura maxima e minima, precipitação, vento e clima.
    Para desenvolver o projeto eu peguei uma basse de dados do histórico de temperatura em Seattle, fiz algumas limpeza nos dados, convertendo tudo para inteiros, e removendo colunas desnecessárias.
    Para testes foi treinado o modelo utilizando as features como as datas, então ele previa as targets em base do dia, o que não faz mto sentido, mas pelo menos consegui fazer funcionar e entender um pouco melhor. O output que eu consegui foi o seguinte:
    -------------precepitation-------------
    6.476272856552249
    ----------------temp_max---------------
    15.294961896316885
    ----------------temp_min---------------
    4.835010297301145
    -----------------wind------------------
    1.5392337456304364
    -------------weather-------------
    194.0
    [[ 1  4  4  0  0]
    [ 1 98 25  1  1]
    [ 1 33 87  0  3]
    [ 0  6  0  2  0]
    [ 0 11  9  0  6]]

    De precipitação ao vento os erros estão sendo medidos pelo rmse (Root Mean Squared Error), que basicamente calcula a distancia do chute com o valor correto.
    Então se RMSE estiver em 0, quer dizer que o modelo é perfeito.

    A partir desse primeiro resultado, irei ajustar todas as features para fazer sentido a maneira em que o modelo vai prever os valores. A ideia inical é:
    Precipitação, temperatura máxima, temperatura mínima e velocidade do vento, vão ser todos previstos de acordo com os ultimos 7 dias, e o clima será previsto em cima desses outros valores
    
    