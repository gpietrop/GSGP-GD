import gplearn.genetic as gp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import sklearn.utils as skutil
import sklearn.metrics as metrics
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas as pd
import os

fset=('add', 'sub', 'mul', 'div')
pop = 100
num_exp = 100

for dname in ["yacht", "bioav", "slump", "toxicity", "ppb", "concrete", "airfoil"]: 
    path_result = "results/" + dname + "/gp/"
    if not os.path.exists(path_result):
        os.mkdir(path_result) 
    path_result = path_result + str(pop) + "/"
    if not os.path.exists(path_result):
        os.mkdir(path_result) 
    
    for i in range(1, num_exp):
        print("\n" + "Dataset : " + dname + "\t" + "Experiment " + str(i))
        train_name = "datasets/" + dname + "/train" + str(i)
        test_name = "datasets/" + dname + "/test" + str(i)
        
        TrainErr_GP, TestErr_GP = [], []
        fname_train = path_result + "results_train_" + str(i) + ".txt"
        fname_test = path_result + "results_test_" + str(i) + ".txt"
        
        res_train = open(fname_train, "w")
        res_test = open(fname_test, "w")
        
        train = pd.read_csv(train_name, sep='\t', header = 2).to_numpy()
        x_train = train[:, :-1]
        y_train = train[:, -1]

        test = pd.read_csv(test_name, sep='\t', header = 2).to_numpy()
        x_test = test[:, :-1]
        y_test = test[:, -1]
        
        max_gen = 200
        sr = gp.SymbolicRegressor(population_size = pop,
                                  generations=1,
                                  function_set=fset,
                                  stopping_criteria=0.01,
                                  p_crossover=0.8, # Probability of performing subtree crossover
                                  p_subtree_mutation=0.1, # Probability of subtree mutation
                                  p_hoist_mutation=0.05, # Small probability of hoist mutation
                                  p_point_mutation=0.05, # Small probability of point mutation
                                  parsimony_coefficient=0.01, # Penalization of large trees
                                  verbose=0, # Set to 1 to obtain the fitness values
                                  random_state=0,
                                  warm_start=True)

        for i in range(0, max_gen+1):
            sr.set_params(generations=i+1)
            sr.fit(x_train, y_train)
            
            y_train_predicted = sr.predict(x_train)
            rmse_train = metrics.mean_squared_error(y_train_predicted, y_train)
            res_train.write(str(rmse_train) + "\n")
            
            y_test_predicted = sr.predict(x_test)
            rmse_test = metrics.mean_squared_error(y_test_predicted, y_test)
            res_test.write(str(rmse_test) + "\n")
            print("train loss : " + str(rmse_train) + "\t" + "test loss : " + str(rmse_test))
            
            
            
            
