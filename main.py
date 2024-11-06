import pandas as pd
from sklearn.linear_model import LinearRegression 
import numpy as np

df = pd.read_csv('data.csv')

colunas_indepentes_x = ["Alunos","TempAmbiente","CompLigados","CapacidadeSala","Horario"]
colunas_dependentes_y = ["TempAr"] 

dados_x = df[colunas_indepentes_x]
dados_y = df[colunas_dependentes_y]

modelo = LinearRegression().fit(dados_x,dados_y)

num_alunos_test = 21
num_temp_ambiente_test = 27
num_comp_ligados_test = 27
num_capacidade_sala_test = 30
num_horario_test = 20

valores_test = np.array([[num_alunos_test,num_temp_ambiente_test,
                          num_comp_ligados_test,num_capacidade_sala_test,
                          num_horario_test]])

predicao = modelo.predict(valores_test)

print(predicao[0][0])