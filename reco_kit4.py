"""
Implementation of Gauss-Newton

"""

import numpy as np
from scipy.io import loadmat
import time

import torch

import matplotlib

matplotlib.use("Agg")

from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt

from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    form,
    dirichletbc,
    locate_dofs_topological,
)

import matplotlib.pyplot as plt
import os
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.interpolate import interpn

from scipy.io import loadmat

from eit_fenicsx.src.forward_model.eit_forward_fenicsx import EIT
from utils import current_method

from scipy.io import loadmat

data = loadmat("KIT4/data_mat_files/datamat_1_1.mat")

B = data["MeasPattern"].T
print(B.shape)

print(B)

print(np.linalg.matrix_rank(B))

L = 16


Injref = data["CurrentPattern"].T

print("Injref: ", Injref.shape)
print(Injref[1, :])

Uel = data["Uel"].T

Bf = np.vstack([B, np.ones(B.shape[-1])])

print(Bf)

print("new rank: ", np.linalg.matrix_rank(Bf))


print("Bf: ", Bf.shape, " Uel: ", Uel.shape)

print("UEL 1 pattern: ", Uel[1, :])

U = []
print(Uel.shape)
for i in range(Uel.shape[0]):
    exU = np.hstack([Uel[i, :], np.array([0])])
    U_sol, res, _, _ = np.linalg.lstsq(Bf, np.hstack([Uel[i, :], np.array([0])]))
    # print(U_sol)
    # print(res)
    U.append(U_sol)

Uel = np.stack(U)

print("UEL 1 pattern: ", Uel[1, :])


print("UEL SHAPE: ", Uel.shape)

z = 0.001 * np.ones(L)  # 1e-6*np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="KIT4_mesh.msh")


# We use piecewise constant functions to approximate the solution
V_sigma = functionspace(solver.omega, ("DG", 0))

# sigma_background = Function(solver.V)
# sigma_background.interpolate(lambda x: 1.3*np.ones_like(x[0]))

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

Lprior = np.load("L_KIT4.npy")

R = Lprior.T @ Lprior
R = torch.from_numpy(R).to("cuda").float()

# We create the ground truth on the same mesh
mesh_pos = np.array(V_sigma.tabulate_dof_coordinates()[:, :2])

# u_all, Usim = solver.forward_solve(sigma_background)
# Usim = np.array(Usim)

# print(Usim.shape)

# print("USIM: ", Usim[1,:])

Umeas = Uel

print(Umeas.shape)

sigma = 1.32

for i in range(161):
    print("STEP: ", i)

    sigma_k = Function(V_sigma)
    sigma_k.x.array[:] = sigma

    u_all, Usim = solver.forward_solve(sigma_k)
    Usim = np.asarray(Usim)  # .flatten()

    time1 = time.time()
    J = solver.calc_jacobian(sigma_k, u_all)

    deltaU = (Usim - Umeas).flatten()

    J = torch.from_numpy(J).float().to("cuda")
    deltaU = torch.from_numpy(deltaU).float().to("cuda")

    time1 = time.time()
    A = J.T @ J
    A = A + 5e-3 * R
    b = J.T @ deltaU

    time1 = time.time()
    delta_sigma = torch.linalg.solve(A, b).cpu().numpy()
    sigma_old = np.copy(sigma)

    step_sizes = np.linspace(0.01, 1, 6)
    losses = []
    for step_size in step_sizes:
        sigma_new = sigma + step_size * delta_sigma

        sigma_new = np.clip(sigma_new, 0.001, 10.0)
        sigmanew = Function(V_sigma)
        sigmanew.x.array[:] = sigma_new

        _, Utest = solver.forward_solve(sigmanew)
        Utest = np.asarray(Utest)  # .flatten()
        losses.append(np.sum((Utest - Umeas) ** 2))

    print(losses)
    step_size = step_sizes[np.argmin(losses)]
    sigma = sigma + step_size * delta_sigma
    # print("NORM DIFFERENCE: ", np.linalg.norm(sigma_gt_tmp.x.array[:] - sigma ))
    if i >= 0:  # i % 20 == 0:
        # diff_plot = Function(V_sigma)
        # diff_plot.x.array[:] = sigma_gt_tmp.x.array[:] - sigma

        sigma_iter = Function(V_sigma)

        fig, ax = plt.subplots(1, 1, figsize=(19, 6))
        im = ax.tripcolor(tri, sigma.flatten(), cmap="jet", shading="flat")
        ax.axis("image")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Prediction")
        fig.colorbar(im, ax=ax)

        # pred = np.array(sigma_gt_tmp.x.array[:]).flatten()
        # im = ax2.tripcolor(tri, pred, cmap='jet', shading='flat')
        # ax2.axis('image')
        # ax2.set_aspect('equal', adjustable='box')
        # ax2.set_title("GT")
        # fig.colorbar(im, ax=ax2)

        # pred = np.array(diff_plot.x.array[:]).flatten()
        # im = ax3.tripcolor(tri, pred, cmap='jet', shading='flat')
        # ax3.axis('image')
        # ax3.set_aspect('equal', adjustable='box')
        # ax3.set_title("|gt - pred|")
        # fig.colorbar(im, ax=ax3)
        plt.savefig(f"tmp/img_{i}.png")
        plt.close()
        # plt.show()
