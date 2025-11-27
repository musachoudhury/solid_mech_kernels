# %%
from mpi4py import MPI
from petsc4py import PETSc
import basix

import numba
from numba import types
import numpy as np

from dolfinx import fem, mesh, cpp, io
import dolfinx.fem.petsc
from dolfinx import default_scalar_type

dtype = default_scalar_type

order = 1
quadrature_degree = 2

msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    n=(1, 1, 1),
    cell_type=mesh.CellType.hexahedron,
)
V = fem.functionspace(msh, ("Lagrange", order, (3,)))

cmap = fem.functionspace(msh, ("Lagrange", 1, (3,)))

u = fem.Function(V)


def symmetry_bc(V):
    sym_bc = []
    domain = V.mesh
    fdim = domain.topology.dim - 1
    for i in range(0, 3):
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, marker=lambda x: np.isclose(x[i], 0.0)
        )
        boundary_dofs = fem.locate_dofs_topological(V.sub(i), fdim, boundary_facets)
        bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V.sub(i))
        sym_bc.append(bc)
    return sym_bc


bcs = symmetry_bc(V)

# facets = mesh.locate_entities_boundary(
#     msh, dim=2, marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 1.0)
# )
# bc = fem.dirichletbc(
#     np.zeros(3, dtype=dtype),
#     fem.locate_dofs_topological(V, entity_dim=2, entities=facets),
#     V=V,
# )

# bcs = [bc]

basix_celltype = getattr(basix.CellType, msh.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, quadrature_degree)

map_c = msh.topology.index_map(msh.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.float64)

# Tabulate basis functions at quadrature points
# (derivative, point, basis fn index, value index)
# phi = V.element.basix_element.tabulate(1, quadrature_points)[0, :, :, 0]

phi = V.element.basix_element.tabulate(1, quadrature_points)

phig = cmap.element.basix_element.tabulate(1, quadrature_points)


gdim = msh.topology.dim
dim = V.element.space_dimension


def detJ(x):
    return abs((x[0, 0] - x[7, 0]) * (x[0, 1] - x[7, 1]) * (x[0, 2] - x[7, 2]))


@numba.njit
def nprint(x):
    """This print method can be used inside Numba's custom kernels
    which are called via external C code (@cfunc).
    """
    print(x)


c_signature = types.void(
    types.CPointer(types.double),  # double *A
    types.CPointer(types.double),  # const double *w
    types.CPointer(types.double),  # const double *c
    types.CPointer(types.double),  # const double *coordinate_dofs
    types.CPointer(types.int32),  # const int *entity_local_index
    types.CPointer(types.uint8),  # const uint8_t *quadrature_permutation
)

print(basix.index(1, 0, 0))
print(basix.index(0, 1, 0))
print(basix.index(0, 0, 1))


# Map to physical reference frame
@numba.cfunc(c_signature, nopython=True)
def tabulate_A(A_, w_, c_, coords_, entity_local_index, quadrature_permutation=None):
    # Wrap pointers as a Numpy arrays
    nQ = len(quadrature_points)
    n_phi = np.int64(dim / gdim)
    A_full = numba.carray(A_, (dim, dim), dtype=dtype)

    J = np.zeros((nQ, gdim, gdim))
    K = np.zeros((nQ, gdim, gdim))

    dphidX = np.zeros((gdim, gdim))
    dphidx_T = np.zeros((gdim, gdim))
    dphidx = np.zeros((gdim, gdim))

    x = numba.carray(coords_, (n_phi, gdim), dtype=dtype)
    scale = abs((x[0, 0] - x[7, 0]) * (x[0, 1] - x[7, 1]) * (x[0, 2] - x[7, 2]))

    for p in range(nQ):
        # nprint(phig[1 : gdim + 1, p, :, 0])
        dPhi_g = phig[1 : gdim + 1, p, :, 0]
        _J = J[p, :, :]
        _J = x.T @ dPhi_g.T
        # nprint(np.linalg.inv(_J))
        K[p, :, :] = np.linalg.inv(_J)

    for k in range(nQ):
        for i in range(n_phi):
            for j in range(n_phi):
                D = np.zeros((6, 6))

                E = 200000.0
                nu = 0.3

                lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                mu = E / (2.0 * (1.0 + nu))

                for p in range(3):
                    for q in range(3):
                        D[p, q] = lmbda
                    D[p, p] = lmbda + 2.0 * mu

                for p in range(3, 6):
                    D[p, p] = mu

                B_T = np.zeros((3, 6))

                B = np.zeros((6, 3))

                for p in range(3):
                    dphidX[p, 0] = phi[1, k, i, 0]
                    dphidX[p, 1] = phi[2, k, i, 0]
                    dphidX[p, 2] = phi[3, k, i, 0]

                dphidx_T[:] = 0

                # Convert to basis derivatives with respect to physical element
                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            # multiply by transpose of K
                            # dphidx_T[p, q] += K[k, r, p] * dphidX[r, q]
                            dphidx_T[p, q] += K[k, r, p] * dphidX[r, q]

                    # B.T matrix
                B_T[0, 0] = dphidx_T[0, 0]
                B_T[1, 1] = dphidx_T[1, 1]
                B_T[2, 2] = dphidx_T[2, 2]

                B_T[1, 3] = dphidx_T[2, 2]
                B_T[2, 3] = dphidx_T[1, 1]

                B_T[0, 4] = dphidx_T[2, 2]
                B_T[2, 4] = dphidx_T[0, 0]

                B_T[0, 5] = dphidx_T[1, 1]
                B_T[1, 5] = dphidx_T[0, 0]

                # Repeat above for B matrix
                for p in range(3):
                    dphidX[p, 0] = phi[1, k, j, 0]
                    dphidX[p, 1] = phi[2, k, j, 0]
                    dphidX[p, 2] = phi[3, k, j, 0]

                dphidx[:] = 0

                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            # multiply by transpose of K
                            # dphidx[p, q] += K[k, r, p] * dphidX[r, q]
                            dphidx[p, q] += K[k, r, p] * dphidX[r, q]

                B[0, 0] = dphidx[0, 0]
                B[1, 1] = dphidx[1, 1]
                B[2, 2] = dphidx[2, 2]

                B[3, 1] = dphidx[2, 2]
                B[3, 2] = dphidx[1, 1]

                B[4, 0] = dphidx[2, 2]
                B[4, 2] = dphidx[0, 0]

                B[5, 0] = dphidx[1, 1]
                B[5, 1] = dphidx[0, 0]

                # C = B.T * D  (3x6 * 6x6 = 3x6)

                C = np.zeros((3, 6))
                for p in range(3):
                    for q in range(6):
                        for r in range(6):
                            C[p, q] += B_T[p, r] * D[r, q]

                # A = C * B    3x6x6x3 = 3x3

                for p in range(3):
                    for q in range(3):
                        for r in range(6):
                            A_full[3 * i + p, 3 * j + q] += (
                                -weights[k] * C[p, r] * B[r, q]
                            )
    # Multiply by detJ
    for i in range(dim):  # row i
        for j in range(dim):  # column j
            A_full[i, j] = scale * A_full[i, j]

    # nprint(A_full)
    # print(A_)


@numba.cfunc(c_signature, nopython=True)
def tabulate_b(b_, w_, c_, coords_, entity_local_index, quadrature_permutation=None):
    # Number of basis functions / quadrature points (for this hex element: 8)
    n_phi = dim // gdim
    nQ = len(quadrature_points)

    # Wrap raw pointers
    b_full = numba.carray(b_, (dim,), dtype=dtype)
    x = numba.carray(coords_, (n_phi, gdim), dtype=dtype)

    # Geometric data
    J = np.zeros((nQ, gdim, gdim), dtype=dtype)
    K = np.zeros((nQ, gdim, gdim), dtype=dtype)

    dphidX = np.zeros((gdim, gdim), dtype=dtype)
    dphidx_T = np.zeros((gdim, gdim), dtype=dtype)

    # N^T, B^T, stress, body force
    N_T = np.zeros((3, 3), dtype=dtype)
    B_T = np.zeros((3, 6), dtype=dtype)
    stress = np.zeros(6, dtype=dtype)  # currently all zeros
    body_force = np.array([1.0, 0.0, 0.0], dtype=dtype)

    # detJ scaling (same as in tabulate_A)
    scale = abs((x[0, 0] - x[7, 0]) * (x[0, 1] - x[7, 1]) * (x[0, 2] - x[7, 2]))

    # Compute Jacobian and inverse at each quadrature point
    for p in range(n_phi):
        dPhi_g = phig[1 : gdim + 1, p, :, 0]
        J[p, :, :] = x.T @ dPhi_g.T
        K[p, :, :] = np.linalg.inv(J[p, :, :])

    # Quadrature loop
    for k in range(nQ):  # quadrature point
        for i in range(n_phi):  # basis function index
            # N^T (3x3): diagonal = scalar shape value
            N_T[:, :] = 0.0
            val_phi = phi[0, k, i, 0]
            for j in range(3):
                N_T[j, j] = val_phi

            # dphi/dX in reference element (same pattern as stiffness kernel)
            for p in range(3):
                dphidX[p, 0] = phi[1, k, i, 0]
                dphidX[p, 1] = phi[2, k, i, 0]
                dphidX[p, 2] = phi[3, k, i, 0]

            # Convert to dphi/dx via transpose of K
            dphidx_T[:, :] = 0.0
            for p in range(3):
                for q in range(3):
                    for r in range(3):
                        dphidx_T[p, q] += K[k, r, p] * dphidX[r, q]

            # Build B^T (3x6)
            B_T[:, :] = 0.0
            B_T[0, 0] = dphidx_T[0, 0]
            B_T[1, 1] = dphidx_T[1, 1]
            B_T[2, 2] = dphidx_T[2, 2]

            B_T[1, 3] = dphidx_T[2, 2]
            B_T[2, 3] = dphidx_T[1, 1]

            B_T[0, 4] = dphidx_T[2, 2]
            B_T[2, 4] = dphidx_T[0, 0]

            B_T[0, 5] = dphidx_T[1, 1]
            B_T[1, 5] = dphidx_T[0, 0]

            # Assemble RHS
            for j in range(3):
                # Internal forces: B^T * sigma * dv (sigma currently zero -> no effect)
                for p in range(6):
                    b_full[3 * i + j] += weights[k] * (B_T[j, p] * stress[p])

                # Body forces: N^T * b * dv
                for p1 in range(3):
                    b_full[3 * i + j] += weights[k] * (N_T[j, p1] * body_force[p1])

                # Traction term (N^T * t) can be added here later if needed
    # nprint(b_full)

    # Multiply by detJ (scale)
    for i in range(dim):
        b_full[i] = scale * b_full[i]

    # nprint(b_full)


active_coeffs = np.array([], dtype=np.int8)

formtype = cpp.fem.Form_float64
cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)
integrals = {
    fem.IntegralType.cell: [
        (0, tabulate_A.address, cells, active_coeffs),
    ]
}

coefficients_A, constants_A = [], []
a = fem.Form(
    formtype(
        [V._cpp_object, V._cpp_object],
        integrals,
        coefficients_A,
        constants_A,
        False,
        [],
        mesh=msh._cpp_object,
    )
)

active_coeffs = np.array([], dtype=np.int8)
coefficients_L, constants_L = [], []
integrals = {fem.IntegralType.cell: [(0, tabulate_b.address, cells, active_coeffs)]}
L = fem.Form(
    formtype(
        [V._cpp_object],
        integrals,
        coefficients_L,
        constants_L,
        False,
        [],
        mesh=msh._cpp_object,
    )
)

A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()

# # Convert to dense format
# A_dense = PETSc.Mat().createDense(A.getSize())
# A.copy(A_dense)

# # Get numpy array
# A_numpy = A_dense.getDenseArray()

# print(A_numpy)
# np.savetxt("A_matrix.txt", A_numpy, fmt="%.6f")
# exit()
b = fem.petsc.assemble_vector(L)
fem.petsc.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
for bc in bcs:
    bc.set(b.array_w)

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("preonly")
ksp.getPC().setType("lu")
# ksp.getPC().setFactorSolverType("mumps")
# A.setOption(PETSc.Mat.Option.SPD, 1)
ksp.setOperators(A)
x = A.getVecRight()
ksp.solve(b, x)

u.x.array[:] = x

print(u.x.array[:])

vtkfile = io.VTKFile(msh.comm, "results/u", "w")
vtkfile.write_function(u, 0.0)
