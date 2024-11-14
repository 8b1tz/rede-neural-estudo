import numpy as np


# não retorna negativo
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def simoid_derivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1], [1], [0]])
# sinapse entrada -> camada oculta
pesos_x1_x2 = np.array([[-0.424, -0.700, -0.961], [0.358, -0.577, -0.469]])
# sinapse camada oculta -> saida
pesos_camada_oculta_para_saida = np.array([[-0.017], [-0.893], [0.148]])
# quantidade de vezes que vamos atualizar os pesos
epocas = 100

camada_entrada = entradas.copy()

for j in range(epocas):
    soma_sinapse_x1_x2 = np.dot(camada_entrada, pesos_x1_x2)
    camada_oculta = sigmoid(soma_sinapse_x1_x2)

    soma_sinapse_oculta = np.dot(camada_oculta, pesos_camada_oculta_para_saida)
    camada_saida = sigmoid(soma_sinapse_oculta)

    erro_camada_saida = saidas - camada_saida
    media_absoluta_saida = np.mean(np.abs(erro_camada_saida))

    derivada_saida = simoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida

    pesos_x1_x2_transpostas = pesos_camada_oculta_para_saida.T
    delta_saida_x_peso = delta_saida.dot(pesos_x1_x2_transpostas)
    delta_camada_oculta = delta_saida_x_peso * simoid_derivada(camada_oculta)

print(np.round(delta_camada_oculta, decimals=3))