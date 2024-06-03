#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable


def gauss_jordan(x: List[float], y: List[float], verbose=0) -> List[float]:
    m, n = x.shape

    augmented_mat: List[float] = np.zeros(shape=(m, n + 1))
    augmented_mat[:m, :n] = x
    augmented_mat[:, m] = y

    np.set_printoptions(precision=2, suppress=True)

    if verbose > 0:
        print('Исходная матрица:')
        print(augmented_mat)
    
    outer_loop: List[List[float]] = [[0, m - 1, 1], [m - 1, 0, -1]]
    
    for d in range(2):
        for i in range(outer_loop[d][0], outer_loop[d][1], outer_loop[d][2]):
            inner_loop: List[List[float]] = [[i + 1, m, 1], [i - 1, -1, -1]]
            for j in range(inner_loop[d][0], inner_loop[d][1], inner_loop[d][2]):
                k: float = (-1) * augmented_mat[j, i] / augmented_mat[i, i]
                temp_row: List[float] = augmented_mat[i, :] * k
                if verbose > 1:
                    print('Используем строку %2i для строки %2i' % (i + 1, j + 1))
                    print('k=%.2f' % k, '*', augmented_mat[i, :], '=', temp_row)
                augmented_mat[j, :] = augmented_mat[j, :] + temp_row
                if verbose > 1:
                    print(augmented_mat)

    for i in range(0, m):
        augmented_mat[i, :] = augmented_mat[i, :] / augmented_mat[i, i]

    if verbose > 0:
        print('Диагональная матрица:')
        print(augmented_mat)
        
    return augmented_mat[:, n]


def L(X: List[float], Y: List[float]) -> Callable[[float], float]:
    if len(X) != len(Y): raise ValueError("Размерность X не совпадает с Y.")

    pairs: Iterable[float] = list(zip(X, Y))
    pairs.sort(key = lambda x: x[0])
    X, Y  = zip(*pairs)

    def polinom(x: float) -> float:
        result: float = 0 
        for i in range(len(X)): 
            term: float = Y[i] 
            for j in range(len(X)): 
                if j != i: 
                    term *= (x - X[j]) / (X[i] - X[j]) 
            result += term 
        return result 

    return polinom


def find_extremums(X: List[float], Y: List[float], lagrange_poly: float) -> List[float]: 
    critical_points: List[float] = [] 
    for i in range(len(X)): 
        x: float = X[i] 
        L_derivative: float = 0 
        for j in range(len(X)): 
            if j != i: 
                term: float = 1 / (x - X[j]) 
                for k in range(len(X)): 
                    if k != i and k != j: 
                        term *= (x - X[k]) / (X[i] - X[k]) 
                L_derivative += term 

        if L_derivative == 0: 
            critical_points.append(x) 

    return critical_points

def main() -> None:
    m_A = np.array([
        [1, 1, 0, 1, -1],
        [1, 2, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 3, -1],
        [-1, 0, 0, -1, 3]
    ])
    m_B = np.array([2, 4, 6, 8, 10])
    m_Y = gauss_jordan(m_A, m_B, 1)

    print(f"Значения y_i: {m_Y}")

    f: Callable[[float], float] = L(m_B, m_Y)
    print(f"Многочлен лагранжа в x_0 = 2: {f(2)}")
    
    e = find_extremums(m_B, m_Y, f(2))
    print(f"Экстремумы: {e}")

    plt.plot(m_B, m_Y)
    #plt.show()

if __name__ == "__main__":
    main()
