
import numpy as np


def twospirals(n_points, n_turns, ts=np.pi, tinc=1, noise=.3):
    """
     Returns the two spirals dataset.
     modificado de:
         https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
    Primeiro gera uma espiral e obtem a segunda espelhando a primeira
    """
    # equação da espiral (coord polares): r = tinc*theta
    # n_points: número de pontos de cada espiral
    # n_turns: número de voltas das espirais
    # ts: ângulo inicial da espiral em radianos
    # tinc: taxa de crescimento do raio em função do ângulo
    # noise: desvio-padrão do ruído

    # Sorteando aleatoriamente pontos da espiral
    n = np.sqrt(np.random.rand(n_points, 1))  # intervalo [0,1] equivale a [0,theta_max]
    # tomar a raiz quadrada ajuda a
    # distribuir melhor os pontos
    ns = (ts) / (2 * np.pi * n_turns)  # ponto do intervalo equivalente a ts radianos
    n = ns + n_turns * n  # intervalo [ns,ns+n_turns] equivalente a [ts, theta_max]
    n = n * (2 * np.pi)  # intervalo [ts, theta_max]

    # Espiral 1
    d1x = np.cos(n) * tinc * n + np.random.randn(n_points, 1) * noise
    d1y = np.sin(n) * tinc * n + np.random.randn(n_points, 1) * noise

    # Espiral 2
    d2x = -np.cos(n) * tinc * n + np.random.randn(n_points, 1) * noise
    d2y = -np.sin(n) * tinc * n + np.random.randn(n_points, 1) * noise

    spirals_points = np.vstack((np.hstack((d1x, d1y)), np.hstack((d2x, d2y))))
    points_labels = np.hstack((np.ones(n_points), np.zeros(n_points)))
    return (spirals_points, points_labels)