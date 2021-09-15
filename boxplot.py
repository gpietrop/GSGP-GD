import matplotlib.pyplot as plt
import seaborn as sns
import os 

dname = "slump"
pop = 50
ep1 = 50 
ep2 = 50
lr = 0.001


num_exp = 50
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
    

sns.set(context='notebook', style='darkgrid')
sns.utils.axlabel(xlabel="Methods", ylabel="Train Error", fontsize=10)
sns.boxplot(data = [TrainErr_GP, TrainErr_HYB, TrainErr_NEW], width=.58, palette = "pastel")
plt.xticks(plt.xticks()[0], ['GP', 'HYB', 'HeH'])
plt.title(dname)
plt.savefig(dire + "/Train_BP.png")
plt.show()
plt.close()

sns.set(context='notebook', style='darkgrid')
sns.utils.axlabel(xlabel="Methods", ylabel="Test Error", fontsize=10)
sns.boxplot(data = [TestErr_GP, TestErr_HYB, TestErr_NEW], width=.58, palette = "pastel")
plt.xticks(plt.xticks()[0], ['GP', 'HYB', 'HeH'])
plt.title(dname)
plt.savefig(dire + "/Test_BP.png")
plt.show()
plt.close()
