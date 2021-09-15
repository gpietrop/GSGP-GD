import matplotlib.pyplot as plt
import seaborn as sns
import os 

dname = "slump"
pop = 50
ep1 = 50 
ep2 = 50
lr = 0.001

epochs = ep1 + ep2

num_exp = 50
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


def plt_loss(selected_sample, case):
    if case == 'train':
        er_gp = TrainErr_GP_List[selected_sample]
        er_hyb = TrainErr_HYB_List[selected_sample]
        er_new = TrainErr_NEW_List[selected_sample]
    if case == 'test':
        er_gp = TestErr_GP_List[selected_sample]
        er_hyb = TestErr_HYB_List[selected_sample]
        er_new = TestErr_NEW_List[selected_sample]
    
    plt.plot(range(epochs), er_gp, 'cornflowerblue', label='GSGP', linewidth=1)
    # plt.plot(range(epochs), er_gp, 'g', label='GSGP', linewidth=0.6, marker = '.')
    plt.plot(range(epochs), er_hyb, 'lightcoral', label='HYB', linewidth=1)
    plt.plot(range(epochs), er_new, 'lightgreen', label='HeH', linewidth=1)
    plt.title(dname + " - " + str(selected_sample))
    plt.xlabel('Iteration')
    plt.ylabel(case + 'RMSE')
    plt.grid(axis="y", linestyle= '--', linewidth=0.5)
    plt.xlim(0, epochs)
    plt.legend()
    
    dire_ = dire + '/' + case
    if not os.path.exists(dire_):
        os.mkdir(dire_)
    
    plt.savefig(dire_ + '/' + case + '_' + str(selected_sample))
    plt.show()
    plt.close()
    
num_exp = 5

for i in range(num_exp):
    plt_loss(i, 'test')





