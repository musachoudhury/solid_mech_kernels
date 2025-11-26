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

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 16),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", order))

u = fem.Function(V)
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
)

basix_celltype = getattr(basix.CellType, msh.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, quadrature_degree)

map_c = msh.topology.index_map(msh.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.float64)

# Tabulate basis functions at quadrature points

phi = V.element.basix_element.tabulate(0, quadrature_points)[0, :, :, 0]

A_ref = np.einsum("k,ki,kj->ij", weights, phi, phi)

dim = V.element.space_dimension


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


# Map to physical reference frame
@numba.cfunc(c_signature, nopython=True)
def tabulate_A(A_, w_, c_, coords_, entity_local_index, quadrature_permutation=None):
    # Wrap pointers as a Numpy arrays
    A = numba.carray(A_, (dim, dim), dtype=dtype)
    coordinate_dofs = numba.carray(coords_, (3, 3))

    x0, y0 = coordinate_dofs[0, :2]
    x1, y1 = coordinate_dofs[1, :2]
    x2, y2 = coordinate_dofs[2, :2]

    # Compute Jacobian determinant and fill the output array with
    # precomputed mass matrix scaled by the Jacobian
    detJ = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    A[:] = detJ * A_ref


@numba.cfunc(c_signature, nopython=True)
def tabulate_b(b_, w_, c_, coords_, entity_local_index, quadrature_permutation=None):
    # Wrap pointers as a Numpy arrays
    b = numba.carray(b_, (dim,), dtype=dtype)
    w = numba.carray(w_, (dim,), dtype=dtype)
    coordinate_dofs = numba.carray(coords_, (3, 3))

    # nprint(w)

    x0, y0 = coordinate_dofs[0, :2]
    x1, y1 = coordinate_dofs[1, :2]
    x2, y2 = coordinate_dofs[2, :2]

    f = np.zeros(dim)

    f[:] = [x0, x1, x2]
    # nprint(w)
    # nprint("***********")

    f_quad_points = phi @ f

    b_ref = phi.T @ (weights * f_quad_points)

    # Compute Jacobian determinant and fill the output array with
    # precomputed mass matrix scaled by the Jacobian
    detJ = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    b[:] = detJ * b_ref


# w = fem.Function(V)
# w.interpolate(lambda x: [x[0] + x[1]])
# print(w.x.array[:])
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

A = fem.petsc.assemble_matrix(a)
A.assemble()
b = fem.petsc.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("preonly")
ksp.getPC().setType("cholesky")
ksp.getPC().setFactorSolverType("mumps")
A.setOption(PETSc.Mat.Option.SPD, 1)
ksp.setOperators(A)
x = A.getVecRight()
ksp.solve(b, x)

u.x.array[:] = x

vtkfile = io.VTKFile(msh.comm, "results/u", "w")
vtkfile.write_function(u, 0.0)
