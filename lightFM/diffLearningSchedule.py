# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:45:46 2016

@author: sunkanggao
"""

import numpy as np
import matplotlib.pyplot as plt

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score

movielens = fetch_movielens()
train, test = movielens['train'], movielens['test']

"""
Experiment
To evaluate the performance of both learning schedules, let's 
create two models and run each for a number of epochs, measuring 
the ROC AUC on the test set at the end of each epoch.
"""
alpha = 1e-3
epochs = 70

adagrad_model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adagrad',
                        user_alpha=alpha,
                        item_alpha=alpha)
                        
adadelta_model = LightFM(no_components=30,
                         loss='warp',
                         learning_schedule='adadelta',
                         user_alpha=alpha,
                         item_alpha=alpha)

adagrad_auc = []
for epoch in range(epochs):
    adagrad_model.fit_partial(train, epochs=1)
    adagrad_auc.append(auc_score(adagrad_model, test).mean())

adadelta_auc = []
for epoch in range(epochs):
    adadelta_model.fit_partial(train, epochs=1)
    adadelta_auc.append(auc_score(adadelta_model, test).mean())
    
x = np.arange(len(adagrad_auc))
plt.figure()
plt.plot(x, np.array(adagrad_auc))
plt.plot(x, np.array(adadelta_auc))
plt.title('loss = wrap')
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()

adagrad_model1 = LightFM(no_components=30,
                        loss='warp-kos',
                        learning_schedule='adagrad',
                        user_alpha=alpha,
                        item_alpha=alpha)
                        
adadelta_model1 = LightFM(no_components=30,
                         loss='warp-kos',
                         learning_schedule='adadelta',
                         user_alpha=alpha,
                         item_alpha=alpha)
                         
adagrad_auc1 = []
for epoch in range(epochs):
    adagrad_model1.fit_partial(train, epochs=1)
    adagrad_auc1.append(auc_score(adagrad_model1, test).mean())

adadelta_auc1 = []
for epoch in range(epochs):
    adadelta_model1.fit_partial(train, epochs=1)
    adadelta_auc1.append(auc_score(adadelta_model1, test).mean())

x = np.arange(len(adagrad_auc1))
plt.figure()
plt.plot(x, np.array(adagrad_auc1))
plt.plot(x, np.array(adadelta_auc1))
plt.title('loss = wrap-kos')
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()

