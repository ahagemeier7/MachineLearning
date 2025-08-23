import pandas as pd

# Lê o arquivo
data = pd.read_csv('titanic.csv')

# Imprime o número de linhas originais
print(f"Número de linhas originais: {len(data)}")

# Conta as duplicatas
num_duplicatas = data.duplicated().sum()
print(f"Número de linhas duplicadas: {num_duplicatas}")

# Remove as duplicatas
data.drop_duplicates(inplace=True)

# Imprime o número de linhas após a remoção
print(f"Número de linhas após remover duplicatas: {len(data)}")