import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 

pop = 50
ep1 = 50
ep2 = 50
lr = 0.1

epochs = ep1 + ep2

num_exp = 70

# for dname in ["yacht", "bioav", "slump", "concrete", "toxicity"]: 
# for dname in ["yacht", "bioav", "slump", "toxicity", "ppb", "concrete", "airfoil"]: 
for dname in ["parkinson"]:
    dire = "results/" + dname + "/" + str(ep1) + "-" + str(ep2) + "-" + str(pop) + "-" + str(lr) 
    
    TrainErr_GP_List, TestErr_GP_List = [], []
    TrainErr_HYB_List, TestErr_HYB_List = [], []
    TrainErr_NEW_List, TestErr_NEW_List = [], []

    for i in range(1, num_exp):
        
        TrainErr_GP, TestErr_GP = [], []
        TrainErr_HYB, TestErr_HYB = [], []
        TrainErr_NEW, TestErr_NEW = [], []
        
        fname_GP = dire + "/results-" + str(i) + "-GP" 
        fname_HYB = dire + "/results-" + str(i) + "-HYB" 
        fname_NEW = dire + "/results-" + str(i) + "-NEW"
        
        if not os.path.exists(fname_GP) or not os.path.exists(fname_HYB) or not             os.path.exists(fname_NEW):
           continue

        res_GP = open(fname_GP)
        res_HYB = open(fname_HYB)
        res_NEW = open(fname_NEW)
        
        errs_GP = res_GP.readlines()
        errs_HYB = res_HYB.readlines()
        errs_NEW = res_NEW.readlines()
        
        for line in range(len(errs_GP)):
            TrainErr_gp = float(errs_GP[line].split()[1])
            TrainErr_GP.append(TrainErr_gp)
            TestErr_gp = float(errs_GP[line].split()[2])
            TestErr_GP.append(TestErr_gp)
        
        for line in range(len(errs_HYB)):
            TrainErr_hyb = float(errs_HYB[line].split()[2])
            TrainErr_HYB.append(TrainErr_hyb)
            TestErr_hyb = float(errs_HYB[line].split()[3])
            TestErr_HYB.append(TestErr_hyb)

        for line in range(len(errs_NEW)):
            TrainErr_new = float(errs_NEW[line].split()[1])
            TrainErr_NEW.append(TrainErr_new)
            TestErr_new = float(errs_NEW[line].split()[2])
            TestErr_NEW.append(TestErr_new)
        
        TrainErr_GP_List.append(TrainErr_GP)
        TestErr_GP_List.append(TestErr_GP)
        
        TrainErr_HYB_List.append(TrainErr_HYB)
        TestErr_HYB_List.append(TestErr_HYB)
        
        TrainErr_NEW_List.append(TrainErr_NEW)
        TestErr_NEW_List.append(TestErr_NEW)


    TrainErr_GP_fin, TestErr_GP_fin = [np.median(x) for x in zip(*TrainErr_GP_List)], [np.median(x) for x in zip(*TestErr_GP_List)] 
    TrainErr_HYB_fin, TestErr_HYB_fin = [np.median(x) for x in zip(*TrainErr_HYB_List)], [np.median(x) for x in zip(*TestErr_HYB_List)]
    TrainErr_NEW_fin, TestErr_NEW_fin = [np.median(x) for x in zip(*TrainErr_NEW_List)], [np.median(x) for x in zip(*TestErr_NEW_List)]


    def plt_loss(case):
        if case == 'train':
            er_gp = TrainErr_GP_fin
            er_hyb = TrainErr_HYB_fin
            er_new = TrainErr_NEW_fin
        if case == 'test':
            er_gp = TestErr_GP_fin
            er_hyb = TestErr_HYB_fin
            er_new = TestErr_NEW_fin
        
        plt.plot(range(epochs), er_gp, 'yellowgreen', label='GSGP', linestyle="dashed",linewidth=1.2)
        plt.plot(range(epochs), er_hyb, 'cornflowerblue', label='HYB', linestyle=(0, (5, 1)), linewidth=1.2)
        plt.plot(range(epochs), er_new, 'slateblue', label='HeH', linestyle="solid", linewidth=1.2)
        plt.xlabel('Iteration')
        plt.ylabel( 'Fitness (' + case + ')')
        plt.grid(axis="y", linestyle= '--', linewidth=0.5)
        plt.xlim(0, epochs)
        plt.legend()
        
        dire_res = "results/z_loss/"
        dire_prop = dire_res + str(ep1) + "-" + str(ep2) + "-" + str(pop) + "-" + str(lr)
        if not os.path.exists(dire_prop):
            os.mkdir(dire_prop)
        
        plt.savefig(dire_prop + '/loss_' + dname + '_' + case)
        # plt.show()
        plt.close()

    plt_loss('test')
    plt_loss('train')





