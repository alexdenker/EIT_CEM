import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.io import loadmat
from dolfinx.fem import Function

from src import EIT, gen_conductivity, GaussNewtonSolverTV



device = "cuda"

L = 16
backCond = 1.0

data = loadmat("KIT4/data_mat_files/datamat_1_0.mat")
Injref = data["CurrentPattern"].T

z = 1e-6 * np.ones(L)
solver = EIT(L, Injref, z, mesh_name="mesh_8_4.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(solver.V_sigma.tabulate_dof_coordinates()[:, :2])

np.random.seed(15) 
sigma_mesh = gen_conductivity(
    mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=backCond
)
sigma_gt_vsigma = Function(solver.V_sigma)
sigma_gt_vsigma.x.array[:] = sigma_mesh

sigma_gt = Function(solver.V)
sigma_gt.interpolate(sigma_gt_vsigma)

# We simulate the measurements using our forward solver
_, U = solver.forward_solve(sigma_gt)
Umeas = np.array(U)

noise_percentage = 0.005
var_meas = (noise_percentage * np.abs(Umeas)) ** 2
delta = 0.005
Umeas = Umeas + delta * np.mean(np.abs(Umeas)) * np.random.normal(
                size=Umeas.shape
            )


Umeas_flatten = np.array(Umeas).flatten()

GammaInv = 1.0 / (var_meas.flatten() + 0.001)

sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = backCond

gauss_newton_solver = GaussNewtonSolverTV(
    solver,
    device=device,
    num_steps=15,
    lamb=0.01,  
    beta=1e-8,  
    GammaInv=GammaInv,
    clip=[0.01, 3.0],
    backCond=backCond
)

sigma_rec_tv = gauss_newton_solver.forward_cg(
    Umeas=Umeas_flatten,
    sigma_init=sigma_init,
    verbose=True,
)

# sigma_rec_tv = Function(solver.V_sigma)
# sigma_rec_tv.x.array[:] = sigma

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

pred = np.array(sigma_gt_vsigma.x.array[:]).flatten()
im = ax1.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax1.axis("image")
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("GT")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.axis("off")

pred = np.array(sigma_rec_tv.x.array[:]).flatten()
im = ax2.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax2.axis("image")
ax2.set_aspect("equal", adjustable="box")
ax2.set_title("Gauss-Newton (TV Prior)")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis("off")

plt.show()
