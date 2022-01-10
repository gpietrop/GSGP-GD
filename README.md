# Combining Geometric Semantic GP with Gradient-descent Optimization

Julia implementation for paper "Combining Geometric Semantic GP with Gradient-descent Optimization", Gloria Pietropolli, Alessia Paoletti, Mauro Castelli, Luca Manzoni, 2021.

### Abstract
Geometric semantic genetic programming (GSGP) is a wellknown variant of genetic programming (GP) where recombination and mutation operators have a clear semantic effect. Both kind of operators have randomly selected parameters that are not optimized by the search process. In this paper we combine GSGP with a well-known gradientbased optimizer, Adam, in order to leverage the ability of GP to operate structural changes of the individuals with the ability of gradient-based
methods to optimize the parameters of a given structure. Two methods, named HYB-GSGP and HeH-GSGP, are defined and compared with GSGP on a large set of regression problems, showing that the use of Adam can improve the performance on the test set. The idea of merging evolutionary computation and gradient-based optimization is a promising way of combining two methods with very different – and complementary – strengths.
