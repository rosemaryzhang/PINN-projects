# PINN-projects
https://www.codegenes.net/blog/pinn-pytorch/
https://arxiv.org/html/2403.00599v1
https://medium.com/data-science/solving-differential-equations-with-neural-networks-afdcf7b8bcc4
https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb

2D Wave Equation

u_tt = c^2 ( u_xx + u_yy)

Residual:
u_tt - c^2 ( u_xx + u_yy ) = f
f = 0 for a perfect solution

Loss:
L = L(PDE) + L(Initial Conditions) + L(B. C.)
L(PDE) = sum i=1-N |f|^2

Features
- Visual representation of wave (e.g. ripple)
- Define neural network
- Compute PDE residual
- Train the PINN

Use tanh for activation function
Use gradient descent algorithm to solve for the loss function

Domain: 
0 < x < 1
0 < y < 1
0 < t < T

Edges at 0
u(0, y, t) = 0
u(1, y, t) = 0
u(x, 0, t) = 0
u(x, 1, t) = 0

Initial conditions:
u = Asin(x*pi)sin(y*pi)cos(omega*t)