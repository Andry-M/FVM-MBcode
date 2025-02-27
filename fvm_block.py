# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 02/04/2025
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Modification of the Finite Volume Method (FVM) solver for the
# stress-strain problem using a block-coupled algorithm instead of a segregated algorithm. 
# -----------------------------------------------------------------------------

from mb_code.fvm import *

# Libraries importation
import numpy as np              # Array manipulation
import scipy.sparse as sps      # Sparse matrices
from typing import Callable     # Type specifications
from tqdm import tqdm           # Loop progression bar
from time import time           # Time measurement  
     
class StressStrain2d_block(StressStrain2d):
    """
        Encapsulate the block-coupled problem of the Finite Volume Method for the stress-strain analysis in 2D.\n
    """
    def __init__(s, mesh : Mesh2d, b_cond : dict, mu : Callable, lambda_ : Callable, f : Callable, 
                 init_Ux : List = None, init_Uy : List = None):
        """
            Parameters:
                - mesh (Mesh2d) : mesh of the domain
                - b_cond (dict) : boundary conditions
                - mu (Callable) : Lame's second parameter function
                - lambda_ (Callable) : Lame's first parameter function
                - f (Callable) : body force function
        """
        super().__init__(mesh=mesh,
                         b_cond=b_cond,
                         mu=mu,
                         lambda_=lambda_,
                         f=f,
                         init_Ux=init_Ux,
                         init_Uy=init_Uy)
        s.n_dim = 2 # Number of dimensions
        
    def stiffness(s):
        """
            Calculate the stiffness matrices A for the x and y components of the displacement.
            The x-axis components are stored first in the matrix (A.shape = (s.n_cells*s.n_dim, s.n_cells*s.n_dim))
            Sparse matrix is stored using the lil_matrix format for modifiyability during the assembly.
            
            Returns:
            - A (scipy.sparse.lil_matrix) : stiffness matrix
        """   
        A = sps.lil_matrix((s.n_cells*s.n_dim, s.n_cells*s.n_dim), dtype=DTYPE)
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells      
            grad_stencil_i = cell.grad_stencil # Gradient estimator 

            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                proj_distance = face['proj_distance'] # Distance between the centroids projected on the normal vector
                weight = face['weight'] # Weight of the cell in the gradient calculation
                grad_stencil_j = s.mesh.cells[j].grad_stencil # Gradient estimator     
                
                A_i = [] # Least square matrix
                A_j = [] # Least square matrix
                grad_stencil = [] # Gradient stencil
                for l in grad_stencil_i: # Loop over the neighboring cells
                    if l in grad_stencil_j:
                        grad_stencil.append(l)
                        other = s.mesh.cells[l]
                        A_i.append(s.mesh.centroids[other.centroid] - s.mesh.centroids[cell.centroid])
                        A_j.append(s.mesh.centroids[other.centroid] - s.mesh.centroids[s.mesh.cells[j].centroid])
                    
                A_i = np.array(A_i, dtype=DTYPE) # Convert to numpy array
                A_j = np.array(A_j, dtype=DTYPE) # Convert to numpy array
                grad_estimator_i = np.linalg.inv(A_i.T @ A_i) @ A_i.T # Calculate the estimator for explicit least square gradient calculation
                grad_estimator_j = np.linalg.inv(A_j.T @ A_j) @ A_j.T # Calculate the estimator for explicit least square gradient calculation

                # x-axis component
                ## Diagonal
                ### x-axis contribution
                coeffx = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normx * normx 
                A[i,i] -= coeffx
                ### y-axis contribution
                coeffy = area * s.mu(xm, ym) / proj_distance * normy * normy
                A[i,i] -= coeffy
                
                # Other cell contribution
                A[i,j] += coeffx + coeffy
                
                ## Transverse terms
                ### x-axis contribution
                coeffx = area * s.mu(xm, ym) * normy
                for index, grad_cell in enumerate(grad_stencil):
                    A[i,i+s.n_cells] -= coeffx * grad_estimator_i[0,index] * weight
                    A[i,grad_cell+s.n_cells] += coeffx * grad_estimator_i[0,index] * weight
                    
                for index, grad_cell in enumerate(grad_stencil):
                    A[i,j+s.n_cells] -= coeffx * grad_estimator_j[0,index] * (1-weight)
                    A[i,grad_cell+s.n_cells] += coeffx * grad_estimator_j[0,index] * (1-weight)

                ### y-axis contribution
                coeffy = area * s.lambda_(xm, ym) * normx
                for index, grad_cell in enumerate(grad_stencil):
                    A[i,i+s.n_cells] -= coeffy * grad_estimator_i[1,index] * weight
                    A[i,grad_cell+s.n_cells] += coeffy * grad_estimator_i[1,index] * weight
                    
                for index, grad_cell in enumerate(grad_stencil):
                    A[i,j+s.n_cells] -= coeffy * grad_estimator_j[1,index] * (1-weight)
                    A[i,grad_cell+s.n_cells] += coeffy * grad_estimator_j[1,index] * (1-weight)

                # y-axis component
                ## Diagonal
                ### y-axis contribution
                coeffy = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normy * normy
                A[i+s.n_cells,i+s.n_cells] -= coeffy
                ### x-axis contribution
                coeffx = area * s.mu(xm, ym) / proj_distance * normx * normx
                A[i+s.n_cells,i+s.n_cells] -= coeffx
                ## Other cell contribution
                A[i+s.n_cells,j+s.n_cells] += coeffx + coeffy
                
                ## Transverse terms
                ### x-axis contribution
                coeffx = area * s.lambda_(xm, ym) * normy
                for index, grad_cell in enumerate(grad_stencil):
                    A[i+s.n_cells,i] -= coeffx * grad_estimator_i[0,index] * weight
                    A[i+s.n_cells,grad_cell] += coeffx * grad_estimator_i[0,index] * weight
                    
                for index, grad_cell in enumerate(grad_stencil):
                    A[i+s.n_cells,j] -= coeffx * grad_estimator_j[0,index] * (1-weight)
                    A[i+s.n_cells,grad_cell] += coeffx * grad_estimator_j[0,index] * (1-weight)

                ### y-axis contribution
                coeffy = area * s.mu(xm, ym) * normx
                for index, grad_cell in enumerate(grad_stencil):
                    A[i+s.n_cells,i] -= coeffy * grad_estimator_i[1,index] * weight
                    A[i+s.n_cells,grad_cell] += coeffy * grad_estimator_i[1,index] * weight
                    
                for index, grad_cell in enumerate(grad_stencil):
                    A[i+s.n_cells,j] -= coeffy * grad_estimator_j[1,index] * (1-weight)
                    A[i+s.n_cells,grad_cell] += coeffy * grad_estimator_j[1,index] * (1-weight)
             
            # Iterate through outer faces (boundaries)
            grad_estimator_i = cell.grad_estimator
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normx, normy = face['normal'] # Normal vector to the face
                proj_distance = face['proj_distance'] # Distance between the centroid and the boundary point projected on the normal vector
                area = face['area'] # Length of the face
                
                # Check if the boundary condition is correctly defined for the boundary
                if face['bc_id'] not in s.b_cond['x'].keys():
                    raise ValueError(f'Boundary condition not complete for boundary {face["bc_id"]}')
                if face['bc_id'] not in s.b_cond['y'].keys():
                    raise ValueError(f'Boundary condition not complete for boundary {face["bc_id"]}')
                
                # Boundary conditions are expressed in the (n, t) basis and have to be transformed to the (x, y) basis
                # x-axis component
                cdt_type_x = s.b_cond['x'][face['bc_id']]['type']
                cdt_type_y = s.b_cond['y'][face['bc_id']]['type']
                
                if cdt_type_x == 'displacement' or cdt_type_x == 'Displacement':
                    ## x-axis contribution
                    A[i,i] -= area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx
                    ## y-axis contribution
                    A[i,i] -= area * s.mu(xb, yb) / proj_distance * normy * normy
                    
                    ### Transverse terms for x-axis
                    ## x-axis contribution
                    coeffx = area * s.mu(xb, yb) * normy
                    for index, grad_cell in enumerate(grad_stencil_i):
                        A[i,i+s.n_cells] -= coeffx * grad_estimator_i[0,index]
                        A[i,grad_cell+s.n_cells] += coeffx * grad_estimator_i[0,index]
                    ## y-axis contribution
                    coeffy = area * s.lambda_(xb, yb) * normx
                    for index, grad_cell in enumerate(grad_stencil_i):
                        A[i,i+s.n_cells] -= coeffy * grad_estimator_i[1,index]
                        A[i,grad_cell+s.n_cells] += coeffy * grad_estimator_i[1,index]
                elif cdt_type_x == 'stress' or cdt_type_x == 'Stress':
                    pass
                    # source term   
                else:
                    raise ValueError(f'Boundary condition type {cdt_type_x} not recognized')
                # y-axis component
                if cdt_type_y == 'displacement' or cdt_type_y == 'Displacement':
                    ## y-axis contribution
                    A[i+s.n_cells,i+s.n_cells] -= area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy
                    ## x-axis contribution
                    A[i+s.n_cells,i+s.n_cells] -= area * s.mu(xb, yb) / proj_distance * normx * normx
                    
                    ### Transverse terms for y-axis
                    ## x-axis contribution
                    coeffx = area * s.lambda_(xb, yb) * normy
                    for index, grad_cell in enumerate(grad_stencil_i):
                        A[i+s.n_cells,i] -= coeffx * grad_estimator_i[0,index]
                        A[i+s.n_cells,grad_cell] += coeffx * grad_estimator_i[0,index]
                    ## y-axis contribution
                    coeffy = area * s.mu(xb, yb) * normx
                    for index, grad_cell in enumerate(grad_stencil_i):
                        A[i+s.n_cells,i] -= coeffy * grad_estimator_i[1,index]
                        A[i+s.n_cells,grad_cell] += coeffy * grad_estimator_i[1,index]
                elif cdt_type_y == 'stress' or cdt_type_y == 'Stress':
                    pass
                    # source term
                else:
                    raise ValueError(f'Boundary condition type {cdt_type_y} not recognized')
        return A.tocsc()
    
    def source_body_force(s):
        """
            Calculate the source term for the body force.
            
            Returns:
            - B (np.array) : source term
        """
        B = np.zeros((s.n_cells*s.n_dim), dtype=DTYPE)
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells
            xi, yi = s.mesh.centroids[cell.centroid] # Coordinates of the centroid
            f = s.f(xi, yi) # Body force at the centroid
            B[i] = - f[0] * cell.volume 
            B[i+s.n_cells] = - f[1] * cell.volume
        
        return B
        
    def grad(s, U : np.array):
        """
            Compute the gradient of the displacement field on the mesh using the least squares method.

            Parameters:
                - U (np.array) : displacement field
        """
        grad_Ux = [] # Gradient of the x-axis displacement field
        grad_Uy = [] # Gradient of the y-axis displacement field
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells
            grad_estimator = cell.grad_estimator # Gradient estimator matrix
            grad_stencil = cell.grad_stencil # Gradient estimator stencil
            
            differences_Ux = []
            for j in grad_stencil:
                differences_Ux.append(U[j] - U[i])
            
            grad_Ux.append(grad_estimator @ np.array(differences_Ux, dtype=DTYPE))
            
            differences_Uy = []
            for j in grad_stencil:
                differences_Uy.append(U[j+s.n_cells] - U[i+s.n_cells])
                
            grad_Uy.append(grad_estimator @ np.array(differences_Uy, dtype=DTYPE))

        grad_Ux = np.array(grad_Ux, dtype=DTYPE)
        grad_Uy = np.array(grad_Uy, dtype=DTYPE)
        grad_U = np.concatenate((grad_Ux, grad_Uy), axis=0)
        
        return grad_U      
  
    def source_boundary(s):
        """
            Calculate the source term for the boundary conditions.

            Returns:
            - B (np.array)
        """    
        B = np.zeros((s.n_cells*s.n_dim))

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells                        
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                proj_distance = face['proj_distance'] # Distance between the centroid and the boundary point projected on the normal vector
                area = face['area'] # Length of the face
                
                # Boundary condition type on both axes
                cdt_type_x = s.b_cond['x'][face['bc_id']]['type']
                cdt_type_y = s.b_cond['y'][face['bc_id']]['type']
                
                # x-axis boundary condition             
                if cdt_type_x == 'displacement' or cdt_type_x == 'Displacement':
                    Unt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Ux_b = Unt_b[0] * normx - Unt_b[1] * normy
                    # y-axis oriented face
                    B[i] -= Ux_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx
                    # x-axis oriented face
                    B[i] -= Ux_b * area * s.mu(xb, yb) / proj_distance * normy * normy                 
                elif cdt_type_x == 'stress' or cdt_type_x == 'Stress':
                    Tnt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Tx_b = Tnt_b[0] * normx - Tnt_b[1] * normy
                    B[i] -= area * Tx_b
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
                
                # y-axis boundary condition             
                if cdt_type_y == 'displacement' or cdt_type_y == 'Displacement':
                    Unt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Uy_b = Unt_b[0] * normy + Unt_b[1] * normx
                    # y-axis oriented face
                    B[i+s.n_cells] -= Uy_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy
                    # x-axis oriented face
                    B[i+s.n_cells] -= Uy_b * area * s.mu(xb, yb) / proj_distance * normx * normx        
                elif cdt_type_y == 'stress' or cdt_type_y == 'Stress':
                    Tnt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Ty_b = Tnt_b[0] * normy + Tnt_b[1] * normx
                    B[i+s.n_cells] -= area * Ty_b 
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        
        return B
   
    def source_correction(s, grad_U : np.array = None):
        """
            Calculate the source term for the grid correction (non-orthogonality + skewness).
                        
            Parameters:
                - grad_U (np.array) : gradient (LSQ method)
            
            Returns:
            - B (np.array) : source term in y-axis
        """  
        B = np.zeros((s.n_cells*s.n_dim))

        # U correction terms
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells            
            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                weight = face['weight'] # Weight of the cell in the gradient calculation
                orth = face['orth_correction'] # Skewness of the face
                skew = face['skew_correction'] # Skewness of the face
                distance = face['distance'] # Distance between the centroids
                tangential = face['tangential'] # Tangential vector to the face

                # Weight the gradient at the face centroid
                grad_Ux_ij = weight * grad_U[i] + (1-weight) * grad_U[j] # Weighted gradient of the x-axis displacement field
                grad_Uy_ij = weight * grad_U[i+s.n_cells] + (1-weight) * grad_U[j+s.n_cells] # Weighted gradient of the y-axis displacement field
                
                # Skewness correction
                skew_correction_x = np.zeros(2)
                if distance[0]!=0:
                    skew_correction_x[0] = 1/distance[0] * (np.dot(skew, grad_U[j] - grad_U[i]))
                if distance[1]!=0:
                    skew_correction_x[1] = 1/distance[1] * (np.dot(skew, grad_U[j] - grad_U[i]))
                grad_Ux_ij += skew_correction_x 
                skew_correction_y = np.zeros(2)
                if distance[0]!=0:
                    skew_correction_y[0] = 1/distance[0] * (np.dot(skew, grad_U[j+s.n_cells] - grad_U[i+s.n_cells]))
                if distance[1]!=0:
                    skew_correction_y[1] = 1/distance[1] * (np.dot(skew, grad_U[j+s.n_cells] - grad_U[i+s.n_cells]))
                grad_Uy_ij += skew_correction_y
                
                # Non-orthogonality correction
                corr_grad_Ux_x = np.dot(grad_Ux_ij, orth/area*normx - tangential*normy)        
                corr_grad_Ux_y = np.dot(grad_Ux_ij, orth/area*normy + tangential*normx)  
                corr_grad_Uy_x = np.dot(grad_Uy_ij, orth/area*normx - tangential*normy)        
                corr_grad_Uy_y = np.dot(grad_Uy_ij, orth/area*normy + tangential*normx)        
                
                ### Correction of x-axis source terms
                ## x-axis component
                # x-axis oriented face
                B[i] -= area * (2*s.mu(xm, ym) + s.lambda_(xm, ym)) * corr_grad_Ux_x * normx
                # y-axis oriented face
                B[i] -= area * s.mu(xm, ym) * corr_grad_Ux_y * normy
                
                ### Correction of y-axis source terms
                ## y-axis component
                # y-axis oriented face
                B[i+s.n_cells] -= area * (2*s.mu(xm, ym) + s.lambda_(xm, ym)) * corr_grad_Uy_y * normy
                # x-axis oriented face
                B[i+s.n_cells] -= area * s.mu(xm, ym) * corr_grad_Uy_x * normx
            
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                area = face['area'] # Length of the face        
                orth = face['orth_correction'] # Skewness of the face
                tangential = face['tangential'] # Tangential vector to the face

                # Non-orthogonality correction
                corr_grad_Ux_x = np.dot(grad_U[i], orth/area*normx - tangential*normy)        
                corr_grad_Ux_y = np.dot(grad_U[i], orth/area*normy + tangential*normx)  
                corr_grad_Uy_x = np.dot(grad_U[i+s.n_cells], orth/area*normx - tangential*normy)        
                corr_grad_Uy_y = np.dot(grad_U[i+s.n_cells], orth/area*normy + tangential*normx)
                
                # x-axis component
                cdt_type = s.b_cond['x'][face['bc_id']]['type']         
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    ### Correction of x-axis source terms
                    ## x-axis component
                    # x-axis oriented face
                    B[i] -= area * (2*s.mu(xb, yb) + s.lambda_(xm, yb)) * corr_grad_Ux_x * normx
                    # y-axis oriented face
                    B[i] -= area * s.mu(xb, yb) * corr_grad_Ux_y * normy 
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # No gradient if stress boundary condition which means no correction
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')

                # y-axis component 
                cdt_type = s.b_cond['y'][face['bc_id']]['type']         
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    ### Correction of y-axis source terms
                    ## y-axis component
                    # y-axis oriented face
                    B[i+s.n_cells] -= area * (2*s.mu(xb, yb) + s.lambda_(xb, yb)) * corr_grad_Uy_y * normy
                    # x-axis oriented face
                    B[i+s.n_cells] -= area * s.mu(xb, yb) * corr_grad_Uy_x * normx
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # No gradient if stress boundary condition which means no correction
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        return B
    
    def solve(s, 
              max_iter : int = 10, 
              apply_grid_correction : bool = False,
              tol_res_norm : float = 1e-15,
              solver : Callable = scipy_sparse_spsolve,
              precond : Callable = lambda *_ : None,
              precond_args : dict = dict()):
        """
            Solve the elastic stress problem for the given mesh and boundary conditions using the Finite Volume Method
            and a block-coupled (actually fully coupled unique system) algorithm.\n
                
            Parameters:
                max_iter (int, default=20) : maximum number of iterations for the unstructured grid correction
                apply_grid_correction (bool, default=False) : apply the unstructured grid correction
                tol_res_norm (float, default=0) : tolerance for the unstructured grid correction (iterative correction) in the norm of the residual
                solver (Callable, default=sps.linalg.spsolve) : sparse solver function to solve the system of equations
                precond (Callable, default= lambda *_ : None) : the preconditionner to use for the solver
                precond_args (dict, default=dict()) : the arguments to pass to the preconditionner
                
            Returns:
                Ux (np.array) x-axis displacement field    
                Uy (np.array) y-axis displacement field
        """
        ### Initialisation ###

        Ux, Uy = s.init_Ux.copy(), s.init_Uy.copy()
        U = np.concatenate((Ux, Uy), axis=0)
        
        if not apply_grid_correction:
            max_iter = 1
                
        # Construct the stiffness matrix and the body force source term
        A = s.stiffness()
        
        # Construct a preconditionner if provided
        M = precond(A, **precond_args) 
        
        B_f = s.source_body_force()
        B_b = s.source_boundary()        
        # Unstructured grid correction
        B_c = np.zeros(s.n_cells*s.n_dim)
        
        B = lambda: B_f + B_b + B_c
        
        # Store the initial statistics
        s.statistics.add('res', [])
        s.statistics.add('res_norm', [])
        s.statistics.add('inner_iterations', [])
        s.statistics.store(
            res = (res := 0),
            res_norm = (res_norm := 0),
            hist_Ux = Ux, hist_Uy = Uy,
            hist_Bx = B()[:s.n_cells],
            hist_By = B()[s.n_cells:] 
        )

        # GRID CORRECTION ITERATIONS
        for step in (pbar := tqdm(range(int(max_iter)))):
            start_time = time()
            grad_U = s.grad(U)

            B_c = s.source_correction(grad_U)
            output = solver(A, B(), x0=U, M=M) # INNER ITERATIONS
            if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                U = output[0]
                inner_statistics = output[1]
            else:
                U = output
                inner_statistics = None
            end_time = time()
            
            # If inner_statistics is a dictionnary
            if inner_statistics is not None:
                if 'info' in inner_statistics.keys():
                    if inner_statistics['info'] > 0:
                        print(f'Iteration {step}: Solver did not converge, try to increase the number of iterations or decrease the tolerance')
            
            # Store the statistics
            s.statistics.store(
                res = (res := residual(A, U, B())),
                res_norm = (res_norm := residual_norm(A, U, B())),
                inner_iterations = inner_statistics,
                hist_Ux = U[:s.n_cells], hist_Uy = U[s.n_cells:],
                hist_Bx = B()[:s.n_cells],
                hist_By = B()[s.n_cells:], 
                outer_iterations = {'time' : end_time-start_time},
            )
            
            # Print the progress of the iterations
            if apply_grid_correction:
                pbar.set_postfix_str(f'Normalized residual for unstructured grid correction: {res_norm:.2e}')
            
            # Check convergence
            if step>1 and res_norm < tol_res_norm:
                break
            elif not apply_grid_correction:
                break
        
        # Store the extra statistics
        s.statistics.add('precond', precond.__name__)
        s.statistics.add('precond_args', precond_args)
        
        return U[:s.n_cells], U[s.n_cells:]

    def compute_stress(s, Ux : np.array, Uy : np.array):
        """
            Compute the stress field on the mesh using the displacement field.\n
            Note that the finite volume methods here assumes a constant stress field in each cell.
            This approximation introduces an error in the stress field mainly at the boundaries of the domain.
            
            Parameters:
                - Ux (np.array) : x-axis displacement field
                - Uy (np.array) : y-axis displacement field
        """
        # Initialize the stress fields       
        Sxx = np.zeros((s.n_cells), dtype=DTYPE)
        Syy = np.zeros((s.n_cells), dtype=DTYPE)
        Sxy = np.zeros((s.n_cells), dtype=DTYPE)
        
        # Compute the gradients of the displacement field
        U = np.concatenate((Ux, Uy), axis=0)
        grad_U = s.grad(U)
        grad_Ux = grad_U[:s.n_cells]
        grad_Uy = grad_U[s.n_cells:]
        
        for i, cell in enumerate(s.mesh.cells):
            centroid = s.mesh.centroids[cell.centroid]
            Sxx[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Ux[i][0]
            Syy[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Uy[i][1]
            Sxy[i] = s.mu(centroid[0], centroid[1]) * (grad_Ux[i][1] + grad_Uy[i][0])
        
        return Sxx, Syy, Sxy

