import numpy as np
from sklearn import datasets


def sigmoid(soma):
    soma = np.clip(soma, -500, 500)
    return 1 / (1 + np.exp(-soma))

def simoid_derivada(sig):
    return sig * (1 - sig)

base = datasets.load_breast_cancer()

camada_entrada = base.data
valores_saida = base.target
saidas = np.empty([569, 1], dtype=int)

for i in range(569):
    saidas[i] = valores_saida[i]

pesos_0 = 2 * np.random.random((30,5)) - 1

pesos_1 = 2 * np.random.random((5,1)) - 1

epocas = 10000
taxa_de_aprendizagem = 0.3
momento = 1

for j in range(epocas):
    soma_sinapse_x1_x2 = np.dot(camada_entrada, pesos_0)
    camada_oculta = sigmoid(soma_sinapse_x1_x2)

    soma_sinapse_oculta = np.dot(camada_oculta, pesos_1)
    camada_saida = sigmoid(soma_sinapse_oculta)

    erro_camada_saida = saidas - camada_saida
    media_absoluta_saida = np.mean(np.abs(erro_camada_saida))

    derivada_saida = simoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida

    pesos_x1_x2_transpostas = pesos_1.T
    delta_saida_x_peso = delta_saida.dot(pesos_x1_x2_transpostas)
    delta_camada_oculta = delta_saida_x_peso * simoid_derivada(camada_oculta)

    camada_oculta_transposta = camada_oculta.T
    pesos_novos_1 = camada_oculta_transposta.dot(delta_saida)
    pesos_camada_oculta_para_saida = (pesos_1 * momento) + (pesos_novos_1 * taxa_de_aprendizagem)

    camada_entrada_transposta = camada_entrada.T
    pesos_novos_0 = camada_entrada_transposta.dot(delta_camada_oculta)
    pesos_0 = (pesos_0 * momento) + (pesos_novos_0 * taxa_de_aprendizagem)
