import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
import time

c = 1

#Neural Network Model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) #x is the input tensor here

#Compute the PDE Residual
def wave_equation_residual(model, x, y, t):
    # Enable gradient computation
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    inputs = torch.cat([x, y, t], dim = 1) 
    u = model(inputs)

    # Compute first order derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs = torch.ones_like(u), create_graph = True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs = torch.ones_like(u), create_graph = True)[0]

    # Compute second derivative
    u_yy = torch.autograd.grad(u_y, y, grad_outputs = torch.ones_like(u_y), create_graph = True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), create_graph = True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs = torch.ones_like(u_t), create_graph = True)[0]

    residual = u_tt - (c**2)*(u_xx + u_yy)
    return residual

#Compute the Boundary Condition Loss
def boundary_values(model, x, y, t):

    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)

    return u

#Compute the Initial Condition loss
def inital_displacement_values(model, x, y, m, n):

    #at t = 0, u = sin(m*pi*x)sin(n*pi*y)
    u_0 = torch.sin(m*x*math.pi) * torch.sin(n*y*math.pi)
    t = torch.zeros(100, 1)

    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)

    error = u - u_0

    return error

def inital_vel_values(model, x, y):

    #start from rest
    t = torch.zeros(100, 1)
    t.requires_grad = True

    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)

    #get velocities
    u_t = torch.autograd.grad(u, t, grad_outputs = torch.ones_like(u), create_graph = True)[0]

    return u_t

#Loss function

#Initialise the PINN
model = PINN(input_size = 3, hidden_size = 64, output_size = 1, num_layers = 4)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Generate sample data
x = torch.rand(100, 1)
y = torch.rand(100, 1)
t = 1.5*torch.rand(100, 1)

for epoch in range(2000):
    optimizer.zero_grad()

    #Compute PDE residual loss
    residual = wave_equation_residual(model, x, y, t)
    pde_loss = torch.mean(residual**2)

    #Compute boundary loss
    #x = 0
    x_0 = boundary_values(model, torch.zeros(100,1), y, t)
    #x = 1
    x_1 = boundary_values(model, torch.ones(100, 1), y, t)
    #y = 0
    y_0 = boundary_values(model, x, torch.zeros(100, 1), t)
    #y = 1
    y_1 = boundary_values(model, x, torch.ones(100, 1), t)

    bc_loss = torch.mean(x_0**2) + torch.mean(x_1**2) + torch.mean(y_0**2) + torch.mean(y_1**2)

    #Compute initial condition loss
    inital_disp = inital_displacement_values(model, x, y, 2, 1)
    initial_vel = inital_vel_values(model, x, y)

    initial_loss = torch.mean(inital_disp**2) + torch.mean(initial_vel**2)

    #Weights
    w_1 = 1
    w_2 = 100
    w_3 = 100

    #Total Loss
    total_loss = w_1*pde_loss + w_2*bc_loss + w_3*initial_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

#Visualise solution with animation

n = 1000 #Number of points of plot

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

X, Y = np.meshgrid(x, y)

#Reshape from 2D to 1D

X_flat = X.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)

#Convert to pytorch tensors
x_tensor = torch.tensor(X_flat, dtype=torch.float32)
y_tensor = torch.tensor(Y_flat, dtype=torch.float32)

#Function to predict surfae at time t0
def pred_surface(model, t0):
    #Create time column
    T_flat = np.full((n**2, 1), t0)

    #Convert to pytorch tensor
    t_tensor = torch.tensor(T_flat, dtype=torch.float32)

    inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)

    #get the predicted displacement values
    model.eval()

    with torch.no_grad():
        u_pred = model(inputs)

    #Convert to numpy
    u_pred = u_pred.numpy()

    #Reshape to 2D
    U = u_pred.reshape(n, n)

    return U

times = np.linspace(0, 10, 20)

#Plot in 3D colormap
FRAMES = 100
FPS = 30

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

surf = 0
tstart = time.time()

def animate(frame):
    ax.clear()

    t0 = frame / FPS

    #get predicted surface
    U = pred_surface(model, t0)

    #plot surface
    surf = ax.plot_surface(X, Y, U, cmap="coolwarm")

    #customise axes
    ax.set_zlim(-0.7, 0.7)

    return surf

ani = animation.FuncAnimation(fig=fig, func=animate, frames=FRAMES, interval=10)
plt.show()