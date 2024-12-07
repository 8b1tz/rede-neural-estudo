from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

rede_neural = MLPClassifier(verbose=True,
                            max_iter=1000,
                            tol=0.00001,
                            activation='logistic',
                            )
rede_neural.fit(entradas, saidas)
rede_neural.predict([[5, 7.2, 5.1, 2.2]])