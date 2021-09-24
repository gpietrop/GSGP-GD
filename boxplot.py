import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 

pop = 50
ep1 = 25 
ep2 = 25
lr = 0.1


for dname in ["yacht", "bioav", "slump", "toxicity",  "airfoil", "concrete", "ppb"]: 
#, "parkinson"]:
    num_exp = 100
    dire = "results/" + dname + "/" + str(ep1) + "-" + str(ep2) + "-" + str(pop) + "-" + str(lr) 

    TrainErr_GP, TestErr_GP = [], []
    TrainErr_HYB, TestErr_HYB = [], []
    TrainErr_NEW, TestErr_NEW = [], []

    for i in range(num_exp):
        fname = dire + "/results-" + str(i) + "-res"
        if not os.path.exists(fname):
            continue
        res = open(fname)
        
        errs = res.readlines()
        str_gp = errs[6]
        str_hyb = errs[9]
        str_new = errs[-1]
        
        TrainErr_gp = float(str_gp.split()[1])
        TestErr_gp = float(str_gp.split()[2])
        
        TrainErr_hyb = float(str_hyb.split()[1])
        TestErr_hyb = float(str_hyb.split()[2])
        
        TrainErr_new = float(str_new.split()[1])
        TestErr_new = float(str_new.split()[2])
        
        TrainErr_GP.append(TrainErr_gp)
        TrainErr_HYB.append(TrainErr_hyb)
        TrainErr_NEW.append(TrainErr_new)
        
        TestErr_GP.append(TestErr_gp)
        TestErr_HYB.append(TestErr_hyb)
        TestErr_NEW.append(TestErr_new)
    
    dire_res = "results/z_boxplot/"
    dire_prop = dire_res + str(ep1) + "-" + str(ep2) + "-" + str(pop) + "-" + str(lr)
    if not os.path.exists(dire_prop):
        os.mkdir(dire_prop)
        
    sns.set(context='notebook', style='whitegrid')
    sns.utils.axlabel(xlabel="Methods", ylabel="Train Error", fontsize=10)
    box_plot = sns.boxplot(data = [TrainErr_GP, TrainErr_HYB, TrainErr_NEW], 
            width=.58, 
            palette = "muted",
            showfliers=False)
    plt.xticks(plt.xticks()[0], ['GP', 'HYB', 'HeH'])
    plt.title(dname)
    
    medians = [np.median(TrainErr_GP), np.median(TrainErr_HYB), np.median(TrainErr_NEW)]
    
    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat_ind in range(len(categories)):
        cat = categories[cat_ind]
        y = medians[cat_ind]
        y = round(y, 2)
        ax.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            # fontweight='bold', 
            size=10,
            color='black',
            bbox=dict(facecolor='white'))
    plt.savefig(dire_prop + "/" + dname + "_Train_BP.png")
#     plt.show()
    plt.close()

    sns.set(context='notebook', style='whitegrid')
    sns.utils.axlabel(xlabel="Methods", ylabel="Test Error", fontsize=10)
    box_plot = sns.boxplot(data = [TestErr_GP, TestErr_HYB, TestErr_NEW], 
            width=.58, 
            palette = "muted",
            showfliers=False)
    plt.xticks(plt.xticks()[0], ['GP', 'HYB', 'HeH'])
    plt.title(dname)
    
    medians = [np.median(TestErr_GP), np.median(TestErr_HYB), np.median(TestErr_NEW)]
    
    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat_ind in range(len(categories)):
        cat = categories[cat_ind]
        y = medians[cat_ind]
        y = round(y, 2)
        ax.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            # fontweight='bold', 
            size=10,
            color='black',
            bbox=dict(facecolor='white'))
    
    plt.savefig(dire_prop + "/" + dname + "_Test_BP.png")
#     plt.show()
    plt.close()
