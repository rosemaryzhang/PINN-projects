# PINN-projects

# 2D Wave Equation

A neural network using PyTorch was created to model the 2D wave equation for a square membrane.

2D Wave Equation: 

$\frac{\partial^2 u}{\partial t^2} = c^2 ( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})$

Residual:

$f = \frac{\partial^2 u}{\partial t^2} - c^2 ( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})$

$f = 0$ for a perfect solution

Loss:
$L_{total} = L_{PDE} + L_{Initial Conditions} + L_{BC}$

where $L_{PDE} = \sum_{i=1}^{N} |f|^2$
