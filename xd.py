from typing import List, Callable

def f(x):
    return (49 / 192 * x**4) - (307 / 48 * x**3) + (2711 / 48 * x**2) - (2453 / 12 * x) + 250

    return s

def L(X: List[float], Y: List[float]) -> Callable[[float], float]:
    if len(X) != len(Y): raise ValueError("Dimensiones diferentes en X e Y.")

    pares: Iterable[float] = list(zip(X, Y))
    pares.sort(key = lambda x: x[0])
    X, Y  = zip(*pares)

    def polinomio(x: float) -> float:
        resultado: float = 0 
        for i in range(len(X)): 
            termino: float = Y[i] 
            for j in range(len(X)): 
                if j != i: 
                    termino *= (x - X[j]) / (X[i] - X[j]) 
            resultado += termino 
        return resultado 

    return polinomio

x = [2, 4, 6, 8, 10]
y = [20, -8, 6, 0, 10] 

print("f(x)", "\tL(x)")
for i in x:
    print(round(f(i), 2), "\t", L(x, y)(i))


