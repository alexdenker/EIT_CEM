import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.io import loadmat
from dolfinx.fem import Function
from PIL import Image 

from src import GaussNewtonSolverTV, EIT, transform_data


device = "cuda"

L = 16
backCond = 1.31


data = loadmat("KIT4/data_mat_files/datamat_1_1.mat")

im_frame = Image.open("KIT4/target_photos/fantom_1_1.jpg")
phantom = np.array(im_frame) / 255.0

data_watertank = loadmat("KIT4/data_mat_files/datamat_1_0.mat")
Uel_background = data_watertank["Uel"].T

B = data["MeasPattern"].T
Injref = data["CurrentPattern"].T
Uel = data["Uel"].T

Uel = transform_data(Uel, B)
Uel_background = transform_data(Uel_background, B)

z = 1e-6*np.ones(L)
solver = EIT(L, Injref, z, mesh_name="mesh_8_4.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(solver.V_sigma.tabulate_dof_coordinates()[:, :2])

delta1 = 0.1  
delta2 = 0.001 
var_meas = (delta1 * np.abs(Uel_background) + delta2 * np.max(np.abs(Uel_background))) ** 2
GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-5))

Umeas_flatten = np.array(Uel).flatten()

sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = backCond

gauss_newton_solver = GaussNewtonSolverTV(solver, 
                                          num_steps=5, 
                                          lamb=0.5,  
                                          beta=1e-6, 
                                          GammaInv=GammaInv,
                                          clip=[0.001, 4.0])

sigma = gauss_newton_solver.forward(
    Umeas=Umeas_flatten,
    sigma_init=sigma_init,
    Uel_background=Uel_background,
    verbose=True,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.imshow(phantom)
ax1.axis("off")

pred = np.array(sigma.x.array[:]).flatten()
im = ax2.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax2.axis("image")
ax2.set_aspect("equal", adjustable="box")
ax2.set_title("Gauss-Newton (TV Prior)")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis("off")

plt.show()
