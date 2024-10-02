import numpy as np


# não retorna negativo
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1], [1], [0]])
# sinapse entrada -> camada oculta
pesos_x1_x2 = np.array([[-0.424, -0.700, -0.961], [0.358, -0.577, -0.469]])
# sinapse camada oculta -> saida
pesos_camada_oculta_para_saida = np.array([[-0.017], [-0.893], [0.148]])
# quantidade de vezes que vamos atualizar os pesos
epocas = 100

camada_entrada = entradas.copy()
print(camada_entrada)
for j in range(epocas):
    soma_sinapse_x1_x2 = np.dot(camada_entrada, pesos_x1_x2)
    camada_oculta = sigmoid(soma_sinapse_x1_x2)

    soma_sinapse_oculta = np.dot(camada_oculta, pesos_camada_oculta_para_saida)
    camada_saida = sigmoid(soma_sinapse_oculta)

print(camada_saida)
