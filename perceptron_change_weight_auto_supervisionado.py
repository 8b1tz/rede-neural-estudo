import numpy as np

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saida_esperada = np.array([0, 0, 0, 1])

# sinapses
pesos = np.array([0.0, 0.0])
taxa_aprendizagem = 0.1


def step_function(soma):
    if soma >= 1:
        return 1
    return 0


def calcula_saida(registro):
    # produto escalar
    s = registro.dot(pesos)
    return step_function(s)


def treinar():
    erro_total = 1
    while erro_total != 0:
        erro_total = 0
        for i in range(len(saida_esperada)):
            saida_calculada = calcula_saida(np.asarray(entradas[i]))
            erro = abs(saida_esperada[i] - saida_calculada)
            erro_total += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
                print("Peso atualizado ", pesos[j])
        print("Total de erros: ", erro_total)
    return 0


treinar()
