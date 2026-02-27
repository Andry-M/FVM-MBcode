# MBcode - Finite Volume Method Solver for small strain solid mechanics 

**INCEPTION** – Investigating New Convergence schEmes for Performance and Time Improvement Of fuel and Neutronics calculations

## Context

The Minimal Benchmarking code has been developed during Andry Guillaume Jean-Marie Monlon Master thesis on
"Investigation into acceleration schemes for nuclear fuel performance mechanics solvers".

It aims at reproducing the behavior of the segregated algorithm used for solid mechanics in OpenFOAM solvers, namely in 
foam-for-nuclear/OFFBEAT. It has been used to quickly implement, test and compare acceleration methods in order to select the most promising
ones and implement them into OFFBEAT later on. 

This code has been partly validated and should not be used for "real" mechanics calculations.

## Overview

MBcode is a Python implementation of the Finite Volume Method (FVM) for solving 2D elastic stress-strain problems. It provides multiple solvers including segregated, block-coupled, and segregated accelerated approaches including Anderson mixing, Newton-Krylov, and machine learning.

## Features

- **Segregated FVM Solver** (`fvm.py`) - Base solver using segregated algorithm
- **Block-Coupled Solver** (`fvm_block.py`) - Fully coupled system approach
- **Anderson Mixing** (`fvm_anderson.py`) - Accelerated convergence using iteration history
- **Newton-Krylov Method** (`fvm_krylov.py`) - Advanced nonlinear solver
- **Machine Learning Integration** (`fvm_ml.py`) - Neural network acceleration
- **Multiple Preconditioners** (`preconditionner.py`) - ILU, Cholesky, SuperLU options
- **Linear Solvers** (`inner_solver.py`) - BiCGSTAB, CG, LGMRES, SpSolve
- **2D Mesh Generation** (`mesher.py`) - Custom mesher for structured/unstructured triangular or quadrilateral grids

## Usage

```python
from mb_code import *

# Material parameters
mu = lambda x, y : 79e9           # Pa
lambda_ = lambda x, y : 116e9     # Pa

# Problem
body_force = lambda x, y: [0, 0] # N/m3 # Gravity
b_cond = {
    'x': {
        'west': {'type': 'displacement', 'value': lambda x, y : [0,0]},
        'east': {'type': 'stress', 'value': lambda x, y : [0,0]},
        'south': {'type': 'stress', 'value': lambda x, y : [0,0]},
        'north': {'type': 'stress', 'value': lambda x, y : [0,0]},
        'hole' : {'type': 'stress', 'value': lambda x, y : [0,0]}
    },
    'y': {
        'west': {'type': 'displacement', 'value': lambda x, y : [0,0]},
        'east': {'type': 'stress', 'value': lambda x, y : [0,-1e4]},  
        'south': {'type': 'stress', 'value': lambda x, y : [0,0]},
        'north': {'type': 'stress', 'value': lambda x, y : [0,0]},
        'hole' : {'type': 'stress', 'value': lambda x, y : [0,0]}
    }
}

# Construct the mesh
width, height = 1.5, 1.
nx, ny = 50, 25
mesh = fvm.BeamMesh2d(width,height,nx,ny) 
mesh.init_quad()
mesh.plot((12,6))

# Build the problem
inner_solver.RTOL = 0.
inner_solver.ATOL = 1e-6
solver = inner_solver.scipy_sparse_cg
precond = preconditionner.spilu
precond_args = {'fill_factor': 10}
problem = fvm.StressStrain2d(mesh, b_cond, mu, lambda_, body_force)

# Solve the problem
Ux, Uy = problem.solve(max_iter=20_000, 
                        tol_res_norm=1e-6,
                        solver = solver,
                        precond = precond,
                        precond_args = precond_args)

# Compute the stresses
Sxx, Syy, Sxy = problem.compute_stress(Ux, Uy)

# Plot the history of convergence
fvm.plot_history_fvm(problem.statistics.trend_x,
                     problem.statistics.trend_y,
                     problem.statistics.res_x,
                     problem.statistics.res_y,
                     problem.statistics.res_norm_x,
                     problem.statistics.res_norm_y)

# Plot the fields
fvm.plot_interpolated_solution(mesh, Ux, Uy, 100, 100, 'displacement field')
fvm.plot_interpolated_solution(mesh, Sxx, Syy, 100, 100, 'stress field')
fvm.plot_interpolated_solution(mesh, Sxy, Sxy, 100, 100, 'shear stress field')
```

## File Structure

- `fvm*.py` - Solver implementations
- `mesher.py` - Mesh generation
- `inner_solver.py` - Linear equation solvers
- `preconditionner.py` - Preconditioning strategies
- `utils.py` - Visualization and post-processing
- `parameters.py` - Global configuration

## Author

Andry Guillaume Jean-Marie Monlon

## Version

1.0