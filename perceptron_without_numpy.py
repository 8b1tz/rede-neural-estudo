entradas = [-1, 7, 5]
pesos = [0.8, 0.1, 0]


def soma(e, p):
    s = 0
    for input, weight in zip(e, p):
        s += input * weight
    return s


def stepFunction(soma):
    if soma > 0:
        return 1
    return 0


r = stepFunction(soma(entradas, pesos))
print(r)
