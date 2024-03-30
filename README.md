# DSMC
Direct Simulation Monte Carlo for homogeneous Boltzmann equation.

$\partial_t f(v) = \frac{1}{Kn} Q(f,f)(v), \qquad \text{ for } v \in \mathbb{R}^2$

with collision kernel given by 

$B(|v - v_*|) = C_\alpha |v - v_*|^\alpha, \qquad \alpha \in [0,1]$.

##
Simulation parameters and initial data are found in *config.py*.
Bird's variable hard sphere algorithm can be foudn in *main.py*.
Accuracy testing for collisional laws and sampling can be found in *tests.py*.


