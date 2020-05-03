import numpy as np
from scipy.special import softmax
from gaussian_classifier import StepGM
from data import twospirals
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


x, y = twospirals(800,2,ts=np.pi,tinc=1,noise=0.5)
x1 = x[y == 0]

sg = StepGM()
sg.fit(x1)
