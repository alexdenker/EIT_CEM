"""
Implementation of a simple Gauss-Newton solver for the CEM. In each iteration we have to solve a linear system of equation. This is solved on the GPU using pytorch.

"""

import numpy as np
from tqdm import tqdm

from dolfinx.fem import Function
from scipy.sparse import csr_array, diags
import time 

from src.forward_model import EIT
from src.reconstructor import Reconstructor


class GaussNewtonSolverTV(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        GammaInv: np.array,
        num_steps: int = 8,
        lamb: float = 0.04,
        beta: float = 1e-6,
        Uel_background: np.array = None,
        clip=[0.001, 3.0],       
        **kwargs,
    ):
        super().__init__(eit_solver)

        self.Ltv = self.construct_tv_matrix()

        self.num_steps = num_steps
        self.lamb = lamb
        self.beta = beta
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = clip

    def construct_tv_matrix(self):
        self.eit_solver.omega.topology.create_connectivity(
            1, 2
        )  # Facet-to-cell connectivity
        self.eit_solver.omega.topology.create_connectivity(
            2, 1
        )  # Cell-to-facet connectivity

        # Number of cells in the mesh
        num_cells = self.eit_solver.omega.topology.index_map(2).size_local

        cell_to_edge = self.eit_solver.omega.topology.connectivity(2, 1)
        # cell_to_edge connects a cell to its border, every cell has always three borders because the cell is a triangle
        # e.g. 0: [4 1 0] "Cell 0 connects to border 4, 1 and 0"

        edge_to_cell = self.eit_solver.omega.topology.connectivity(1, 2)
        # edge_to_cell connects every border to a cell, here each border always has 1 or 2 cells
        # e.g. 7011: [4593 4600 ] "Border 7011 is part of cell 4593 and 4600" => This means that cells 4593 and 4600 are connected

        rows = []
        cols = []
        data = []

        row_idx = 0
        for cell in range(num_cells):
            # find borders of cell
            adjacent_edges = cell_to_edge.links(cell)
            for edge in adjacent_edges:
                # find cells connected to this border
                adjacent_cells = edge_to_cell.links(edge)
                if (
                    len(adjacent_cells) > 1
                ):  # only look at the parts where we have two cells
                    rows.append(row_idx)
                    rows.append(row_idx)
                    cols.append(adjacent_cells[0])
                    cols.append(adjacent_cells[1])
                    data.append(1)
                    data.append(-1)

                    row_idx += 1

        return csr_array((data, (rows, cols)), shape=(row_idx, num_cells))
        # CUDA does currently not really support CSR tensors
        # return torch.sparse_csr_tensor(torch.tensor(rows), torch.tensor(cols), torch.tensor(data), dtype=torch.float64,size=(row_idx, num_cells))

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        debugging = kwargs.get("debugging", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            sigma_init = Function(self.eit_solver.V_sigma)
            sigma_init.x.array[:] = self.backCond

        sigma = sigma_init.x.array[:]

        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma

                if debugging:
                    print("Forward Solve...")
                    t1 = time.time()
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if debugging:
                    print("Create Jacobian...")
                    t1 = time.time()
                J = self.eit_solver.calc_jacobian(sigma_k, u_all)
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas

                if debugging:
                    print("Create A and b...")
                    t1 = time.time()
                A = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), J])
                b = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), deltaU])

                L_sigma = np.abs(self.Ltv @ np.array(sigma_k.x.array[:])) ** 2
                eta = np.sqrt(L_sigma + self.beta)
                E = diags(1 / eta)
                LTEL = self.Ltv.T @ E @ self.Ltv
                A = A + self.lamb * LTEL
                b = b - self.lamb * LTEL @ sigma_k.x.array[:]

                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                delta_sigma = np.linalg.solve(A, b)

                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma

                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()

                    tv_value = self.lamb * np.sqrt(((self.Ltv @ sigma_new) ** 2) + self.beta).sum()

                    meas_value = 0.5 * np.sum((np.diag(self.GammaInv) @ (Utest - Umeas)) ** 2)
                    losses.append(meas_value + tv_value)

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size * delta_sigma

                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:]) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco


    def forward_cg(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        debugging = kwargs.get("debugging", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            sigma_init = Function(self.eit_solver.V_sigma)
            sigma_init.x.array[:] = self.backCond

        sigma = sigma_init.x.array[:]

        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma

                if debugging:
                    print("Forward Solve...")
                    t1 = time.time()
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if debugging:
                    print("Create Jacobian...")
                    t1 = time.time()
                J = self.eit_solver.calc_jacobian(sigma_k, u_all)
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas
                
                if debugging:
                    print("Solve System with CG...")
                    t1 = time.time()
                L_sigma = (self.Ltv @ sigma)**2
                eta = np.sqrt(L_sigma + self.beta)
                E = diags(1 / eta)
                LTET = self.Ltv.T @ E @self.Ltv
                def Afwd(x):
                    return np.linalg.multi_dot([J.T, np.diag(self.GammaInv), J, x]) + self.lamb * LTET @ x

                b = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), deltaU])
                b = b - self.lamb * self.Ltv.T @ E @ self.Ltv @ sigma

                delta_sigma = conjugate_gradient(Afwd, b, max_iter=30, verbose=debugging) #np.linalg.solve(A, b)
                
                if debugging:
                    t2 = time.time()
                    print(f"Took {t2-t1}s")

                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma

                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()

                    tv_value = self.lamb * np.sqrt(((self.Ltv @ sigma_new) ** 2) + self.beta).sum()

                    meas_value = 0.5 * np.sum((np.diag(self.GammaInv) @ (Utest - Umeas)) ** 2)
                    losses.append(meas_value + tv_value)

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size * delta_sigma

                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:]) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco
    




def conjugate_gradient(
    A,
    b,
    max_iter: float = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
    init=None,
    verbose=False,
):
    """
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param Callable A: Linear operator as a callable function, has to be square!
    :param input tensor of shap
    :param int max_iter: maximum number of CG iterations
    :param float tol: absolute tolerance for stopping the CG algorithm.
    :param float eps: a small value for numerical stability
    :param  init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: :math:`x` verifying :math:`Ax=b`.

    """

    if init is not None:
        x = init
    else:
        x = np.zeros_like(b)

    r = b - A(x)
    p = r
    rsold = np.dot(r, r).real
    flag = True
    tol = np.dot(b, b).real * (tol**2)
    for _ in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (np.dot(p, Ap) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = np.dot(r, r).real
        if rsnew < tol:
            if verbose:
                print("CG Converged at iteration", _)
            flag = False
            break
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

    if flag and verbose:
        print(f"CG did not converge: Residual {rsnew}")

    return x
