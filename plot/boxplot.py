import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 

pop = 50
ep1 = 50
ep2 = 50
lr = 0.1


# for dname in ["yacht", "bioav", "slump", "concrete", "toxicity"]: 
for dname in ["yacht", "bioav", "slump", "toxicity",  "airfoil", "concrete", "ppb"]:
# for dname in ["parkinson"]:
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
    
    dire_res = "results/boxplot/"
    dire_prop = dire_res + str(ep1) + "-" + str(ep2) + "-" + str(pop) + "-" + str(lr)
    if not os.path.exists(dire_prop):
        os.mkdir(dire_prop)
        
    sns.set(context='notebook', style='whitegrid')
    sns.utils.axlabel(xlabel = "fitness (train)", ylabel=None, fontsize=10)
    box_plot = sns.boxenplot(data = [TrainErr_GP, TrainErr_HYB, TrainErr_NEW], 
        # width=.58, 
        # palette = "muted",
        palette = ['yellowgreen', 'cornflowerblue', 'slateblue'],
        orient = "h",
        k_depth=5,
        )
        # medianprops=dict(color="darkred", alpha=0.8),
        # showfliers=False)

    for i,box in enumerate(box_plot.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')

        # iterate over whiskers and median lines
        # for j in range(5*i,5*(i+1)):
        #     box_plot.lines[j].set_color('black')

    plt.yticks(plt.yticks()[0], ['GP', 'HYB', 'HeH'])
        
    medians = [np.median(TrainErr_GP), np.median(TrainErr_HYB), np.median(TrainErr_NEW)]
        
    plt.savefig(dire_prop + "/" + dname + "_Train_BP_fliers.png")
    # plt.show()
    plt.close()

    sns.set(context='notebook', style='whitegrid')
    sns.utils.axlabel(xlabel="fitness (test)", ylabel=None, fontsize=10)
    box_plot = sns.boxenplot(data = [TestErr_GP, TestErr_HYB, TestErr_NEW], 
                       # width=.5, 
                       palette = ['yellowgreen', 'cornflowerblue', 'slateblue'],
                       orient = "h",
                       k_depth = 5, 
                       )
                       # medianprops=dict(color="darkred", alpha=0.8),
                       # showfliers = False)
                       # flierprops = dict(markerfacecolor = '0.50', markersize = 2))
                           # flierprops = dict(markerfacecolor = '0.50', markersize = 2))

    for i,box in enumerate(box_plot.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')

    plt.yticks(plt.yticks()[0], ['GSGP', 'HYB', 'HeH'])

    plt.savefig(dire_prop + "/" + dname + "_Test_BP_fliers.png")
    # plt.show()
    plt.close()
