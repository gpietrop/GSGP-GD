# Combining Geometric Semantic GP with Gradient-descent Optimization

Julia implementation for paper "Combining Geometric Semantic GP with Gradient-descent Optimization", Gloria Pietropolli, Alessia Paoletti, Mauro Castelli, Luca Manzoni, 2021.

### Abstract
Geometric semantic genetic programming (GSGP) is a wellknown variant of genetic programming (GP) where recombination and mutation operators have a clear semantic effect. Both kind of operators have randomly selected parameters that are not optimized by the search process. In this paper we combine GSGP with a well-known gradientbased optimizer, Adam, in order to leverage the ability of GP to operate structural changes of the individuals with the ability of gradient-based
methods to optimize the parameters of a given structure. Two methods, named HYB-GSGP and HeH-GSGP, are defined and compared with GSGP on a large set of regression problems, showing that the use of Adam can improve the performance on the test set. The idea of merging evolutionary computation and gradient-based optimization is a promising way of combining two methods with very different – and complementary – strengths.

## Instructions

Code runs with python 3.8.5 and Julia 1.4.1 on Ubuntu 20.04.
To run the code, enter the following command:

```bash
julia gsgp_Adam.jl 
```
that will return the fitness results for the 100 runs performed for the following methods: 
- __GSGP__
- __HYB-GSGP__
- __HeH-GSGP__

and save them in the `results` folder.

The script automatically perform the experiments for all the considered benchmark problems, that are: 
- _human oral bioavaibility_
- _median lethal dose_
- _protein-plasma binding level_
- _yacht hydrodynamics_
- _concrete slump_
- _concrete compressive strenght_
- _airfoil self-noise_

that are saved in the `dataset` folder.

The code to reproduce the plot of the paper is contained in the folder `plot`, it is sufficient to run:
```bash
python3 boxplot.py 
```

For example, for the __airfoil__ benchmark problem, the comparison of the fitness results obtained for training and testing set for different methods analyzed is: 



<img src="/img/airfoil_Train_BP.png" width="350" height="300"> <img src="/img/airfoil_Test_BP.png" width="350" height="300"> 
