# Machine Learning Projects

<<<<<<< HEAD
Este repositório reúne projetos desenvolvidos para aprofundar o entendimento sobre algoritmos de aprendizado de máquina. Cada projeto inclui uma breve descrição do modelo utilizado, desafios encontrados durante o desenvolvimento e observações relevantes para aplicações futuras.
=======
Nesse repositório contém alguns projetos em que ajudam a entender melhor o funcionamento do aprendizado de maquina junto com uma breve descrição de cada modelo/projeto, erro encontrado, e uma sobre alguns métodos importantes que possam me ajudar em problemas futuros.
>>>>>>> 5eb42e745c163beae77f676cdb0b4f24bbf62862

---

<<<<<<< HEAD
## Projetos
=======
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
         
    Esse código transforma os valores que tem na base de dados em números entre 0 e 1. Isso serve para fazer com que a "IA" consiga fazer toda a correlação dos dados conseguindo "Comparar" eles.
    Um exemplo para entender isso, seria o caso de Age e Fare, se você for calcular a "distância" entre dois passageiros (como o KNN faz), a Tarifa, por ter valores muito maiores, vai dominar completamente o cálculo. A Idade, com seus valores menores, quase não fará diferença. É como tentar comparar a altura de uma pessoa (em metros) com o peso de um elefante (em quilos) diretamente.
>>>>>>> 5eb42e745c163beae77f676cdb0b4f24bbf62862

### 1. Previsão de Sobrevivência no Titanic — K-Nearest Neighbors (KNN)

Este projeto tem como objetivo prever a sobrevivência de passageiros do Titanic utilizando o algoritmo K-Nearest Neighbors (KNN). O foco principal é o aprendizado prático das bibliotecas `matplotlib`, `scikit-learn` e `seaborn`.

<<<<<<< HEAD
**Principais etapas do projeto:**
- **Limpeza de Dados:** Tratamento de valores nulos e remoção de colunas irrelevantes.
- **Separação de Features e Target:** Os dados são divididos em variáveis independentes (X) e dependentes (y), facilitando o treinamento do modelo.
- **Normalização:** Utilização do `MinMaxScaler` para transformar os dados em uma escala de 0 a 1, garantindo que todas as variáveis tenham o mesmo peso no cálculo das distâncias.
    ```python
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    ```
    *Observação:* A normalização é fundamental para evitar que variáveis com escalas maiores dominem o cálculo de distância do KNN.
- **Validação de Modelos:** Teste de diferentes parâmetros para selecionar o modelo com melhor acurácia.
- **Predição e Avaliação:** Realização de previsões e visualização da matriz de confusão com `matplotlib`.

**Desafio encontrado:**  
Durante o desenvolvimento, foi identificado que o dataset apresentava um viés: todos os homens morriam e todas as mulheres sobreviviam. Isso levou o modelo a tomar decisões baseadas apenas no gênero, resultando em uma acurácia artificialmente alta (100%). O problema foi solucionado após a análise crítica dos dados.

---

### 2. Previsão do Tempo — Random Forest Classifier

O objetivo deste projeto é prever o clima utilizando o algoritmo Random Forest Classifier, com base em dados históricos de Seattle.

**Principais etapas do projeto:**
- **Limpeza e Preparação dos Dados:** Conversão de variáveis para o formato adequado e remoção de colunas desnecessárias.
- **Seleção de Features:** Utilização de variáveis como data, precipitação, temperatura máxima e mínima, e velocidade do vento para prever o clima.
- **Treinamento e Avaliação:** O modelo é treinado e avaliado para medir sua capacidade preditiva.

---

## Estrutura da Pasta `MLAlgorithms`

A pasta `MLAlgorithms` contém implementações e tutoriais de diversos algoritmos clássicos de aprendizado de máquina, organizados da seguinte forma:



## Estrutura da Pasta `MLAlgorithms`

A pasta `MLAlgorithms` contém implementações e tutoriais de diversos algoritmos clássicos de aprendizado de máquina, organizados da seguinte forma:

- **Teste/**  
  Pasta reservada para scripts de teste e experimentação dos algoritmos implementados.

- **Tutorial/**  
  Contém subpastas, cada uma dedicada a um algoritmo específico, com implementações, explicações e exemplos práticos:

    - **DecisionTree/**  
      - `DecisionTree.py`: Implementação de uma Árvore de Decisão do zero.
      - Projeto: Demonstração da construção e uso de árvores de decisão para classificação de dados simples.
      - `DecisionTrees.txt`: Explicação teórica e exemplos de aplicação.

    - **KNN/**  
      - `KNN.py`: Implementação do algoritmo K-Nearest Neighbors.
      - Projeto: Classificação de pontos em conjuntos de dados sintéticos para ilustrar o funcionamento do KNN.
      - `KNearestNeighbors.txt`: Material explicativo sobre o algoritmo.

    - **LinearRegression/**  
      - `LinearRegression.py`: Implementação de Regressão Linear.
      - Projeto: Ajuste de uma reta a dados simulados para prever valores contínuos.
      - `LinearRegression.txt`: Explicação teórica e exemplos.

    - **LogisticRegression/**  
      - `LogisticRegression.py`: Implementação de Regressão Logística.
      - Projeto: Classificação binária em conjuntos de dados simulados para ilustrar a separação de classes.
      - `LogisticRegression.txt`: Material explicativo sobre o algoritmo.

    - **Naive Bayes/**  
      - `NaiveBayes.py`: Implementação do classificador Naive Bayes.
      - Projeto: Classificação de dados sintéticos para demonstrar o cálculo das probabilidades e a tomada de decisão do modelo.
      - `NaiveBayes.txt`: Explicação teórica e exemplos de uso.

    - **RandomForests/**  
      - `RandomForest.py`: Implementação simplificada do algoritmo Random Forest.
      - Projeto: Demonstração da combinação de múltiplas árvores de decisão para melhorar a acurácia em tarefas de classificação.
      - `RandomForests.txt`: Material explicativo sobre o funcionamento do ensemble.

Cada subpasta contém arquivos `.py` com a implementação do algoritmo e arquivos `.txt` com explicações teóricas e exemplos de uso prático.
=======
-Previsão do tempo - RandomForestClassifier
    Para esse projeto, a ideia inicial era criar um modelo para prever o clima.
    Para desenvolver o projeto eu peguei uma basse de dados do histórico de temperatura em Seattle, fiz algumas limpeza nos dados, convertendo tudo para inteiros, e removendo colunas desnecessárias.
    Para testes foi treinado o modelo utilizando as features de data, precipitação, temperatura maxima e minima e velocidade do vento, para prever o clima


  
>>>>>>> 5eb42e745c163beae77f676cdb0b4f24bbf62862
