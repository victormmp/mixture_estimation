import numpy as np
from scipy.special import softmax
from gaussian_classifier import ExpectationMaximization, GaussianMixture, GreedyLearningGM
from data import twospirals
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_curve

x, y = twospirals(800,2,ts=np.pi,tinc=1,noise=0.5)
x1 = x[y == 0]

greedy = GreedyLearningGM(20)
greedy.fit(x1)