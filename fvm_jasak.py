# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 28/11/2024
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Finite Volume Method (FVM) segregated solver for the stress-strain problem
# -----------------------------------------------------------------------------

from mb_code.mesher import *
from mb_code.utils import *
from mb_code.inner_solver import scipy_sparse_spsolve
from mb_code import inner_solver, parameters
from mb_code.parameters import DTYPE

# Libraries importation
import numpy as np                # Array manipulation
import scipy.sparse as sps        # Sparse matrices
from typing import Callable       # Type specifications
from tqdm import tqdm             # Loop progression bar
from time import time             # Time measurement  

class Statistics():
    """
        Encapsulate the statistics of the solver.\n
    """
    def __init__(s, **kwargs):
        """
            Initialize the following quantities:
            - 'trend_x' (List[float]) : trend of the x-axis field component
            - 'trend_y' (List[float]) : trend of the y-axis field component
            - 'res_x' (List[float]) : absolute residuals for the x-axis
            - 'res_y' (List[float]) : absolute residuals for the y-axis
            - 'res_norm_x' (List[float]) : normalized residuals for the x-axis (OpenFOAM-like)
            - 'res_norm_y' (List[float]) : normalized residuals for the y-axis (OpenFOAM-like)
            - 'on_fly_res_x' (List[float]) : absolute residuals between iterations for the x-axis
            - 'on_fly_res_y' (List[float]) : absolute residuals between iterations for the y-axis
            - 'on_fly_res_norm_x' (List[float]) : normalized residuals between iterations for the x-axis
            - 'on_fly_res_norm_y' (List[float]) : normalized residuals between iterations for the y-axis
            - 'hist_Ux' (List[np.array]): the history of the x-axis displacement field
            - 'hist_Uy' (List[np.array]): the history of the y-axis displacement field
            - 'hist_Bx' (List[dict]): the history of the x-axis source vectors
            - 'hist_By' (List[dict]): the history of the y-axis source vectors
            - 'outer_iterations' (List[?]): the statistics of the outer iterations
            - 'inner_iterations' (List[?]): the statistics of the inner iterations 
        """
        # Initialize the statistics that are provided
        for key, value in kwargs.items():
            setattr(s, key, value)
        
        # Initialize the statistics not provided in kwargs
        for key in ['trend_x', 'trend_y','res_x', 'res_y', 'res_norm_x', 'res_norm_y', 'on_fly_res_x', 'on_fly_res_y',
                    'on_fly_res_norm_x', 'on_fly_res_norm_y', 'hist_Ux', 'hist_Uy', 'hist_Bx', 'hist_By',
                    'outer_iterations', 'inner_iterations']:
            if not hasattr(s, key):
                setattr(s, key, [])
        
    def store(s, **kwargs):
        """
            Append the provided statistics to the existing ones.
        """
        for key, value in kwargs.items():
            if key in s.__dict__.keys():
                s.__dict__[key].append(value)
            else:
                raise ValueError(f'The key {key} is not recognized in the statistics dictionary')
    
    def add(s, key, value):
        """
            Add a new key-value pair to the statistics.
        """
        setattr(s, key, value)
            
class StressStrain2d():
    """
        Encapsulate the problem of the Finite Volume Method for the stress-strain analysis in 2D.\n
    """
    def __init__(s, mesh : Mesh2d, b_cond : dict, mu : Callable, lambda_ : Callable, f : Callable,
                 init_Ux : List = None, init_Uy : List = None):
        """
            Parameters:
                - mesh (Mesh2d) : mesh of the domains
                - b_cond (dict) : boundary conditions
                - mu (Callable) : Lame's second parameter function
                - lambda_ (Callable) : Lame's first parameter function
                - f (Callable) : body force function
        """
        s.mesh = mesh
        s.n_cells = len(mesh.cells)
        s.b_cond = b_cond
        s.mu = mu
        s.lambda_ = lambda_  
        s.f = f
        s.statistics = Statistics()
        
        # Initialize the displacement fields
        def initialize_displacement(init_U):
            if init_U is not None:
                return np.array(init_U, dtype=DTYPE)
            else:
                return np.zeros((s.n_cells), dtype=DTYPE)
        
        s.init_Ux = initialize_displacement(init_Ux)
        s.init_Uy = initialize_displacement(init_Uy)
        
    def stiffness(s):
        """
            Calculate the stiffness matrices Ax and Ay for the x and y components of the displacement.
            
            Sparse matrices are stored using the lil_matrix format for modifiyability during the assembly.
            
            Returns:
            - Ax (scipy.sparse.lil_matrix) : stiffness matrix for the x component
            - Ay (scipy.sparse.lil_matrix) : stiffness matrix for the y component
        """   
        Ax = sps.lil_matrix((s.n_cells, s.n_cells), dtype=DTYPE)
        Ay = sps.lil_matrix((s.n_cells, s.n_cells), dtype=DTYPE)

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells          
            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                proj_distance = face['proj_distance'] # Distance between the centroids projected on the normal vector
                area = face['area'] # Length of the face

                # x-axis component
                ## Diagonal
                ### x-axis contribution
                coeffx = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normx * normx 
                Ax[i,i] += coeffx
                ### y-axis contribution
                coeffy = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normy * normy
                Ax[i,i] += coeffy
                ## Off-diagonal
                Ax[i,j] -= coeffx + coeffy

                # y-axis component
                ## Diagonal
                ### y-axis contribution
                coeffy = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normy * normy
                Ay[i,i] += coeffy
                ### x-axis contribution
                coeffx = area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) / proj_distance * normx * normx
                Ay[i,i] += coeffx
                ## Off-diagonal
                Ay[i,j] -= coeffx + coeffy
             
            # Iterate through outer faces (boundaries)
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
                cdt_type = s.b_cond['x'][face['bc_id']]['type']
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    ## x-axis contribution
                    Ax[i,i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx
                    ## y-axis contribution
                    Ax[i,i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy
                    ## No off-diagonal contribution as it is fixed by the Dirichlet condition in the source vector
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # Stress boundary conditions do not contribute to the stiffness matrix
                else:
                    raise ValueError(f'Boundary condition type {cdt_type} not recognized')
                # y-axis component
                cdt_type = s.b_cond['y'][face['bc_id']]['type']
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    ## y-axis contribution
                    Ay[i,i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy
                    ## x-axis contribution
                    Ay[i,i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx
                    ## No off-diagonal contribution as it is fixed by the Dirichlet condition in the source vector
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # Stress boundary conditions do not contribute to the stiffness matrix
                else:
                    raise ValueError(f'Boundary condition type {cdt_type} not recognized')
        return Ax.tocsc(), Ay.tocsc()
    
    def source_body_force(s):
        """
            Calculate the source term for the body force.
            
            Returns:
            - Bx (np.array) : source term for the x component
            - By (np.array) : source term for the y component
        """
        Bx = np.zeros((s.n_cells), dtype=DTYPE)
        By = np.zeros((s.n_cells), dtype=DTYPE)
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells
            xi, yi = s.mesh.centroids[cell.centroid] # Coordinates of the centroid
            f = s.f(xi, yi) # Body force at the centroid
            Bx[i] = f[0] * cell.volume 
            By[i] = f[1] * cell.volume
        
        return Bx, By
        
    def grad(s, U : np.array):
        """
            Compute the gradient of the displacement field on the mesh using the least squares method.

            Parameters:
                - U (np.array) : displacement field component
        """
        grad_U = [] # Gradient of the x-axis displacement field
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells
            grad_estimator = cell.grad_estimator # Gradient estimator matrix
            grad_stencil = cell.grad_stencil # Gradient estimator stencil
            
            differences = []
            for j in grad_stencil:
                differences.append(U[j] - U[i])
            
            grad_U.append(grad_estimator @ np.array(differences, dtype=DTYPE))
        
        grad_U = np.array(grad_U, dtype=DTYPE)
        
        return grad_U       
    
    def source_transverse_x(s, grad_Ux : np.array = None, grad_Uy : np.array = None):
        """
            Calculate the source term for the transverse contribution for the x-axis component.
                        
            Parameters:
                - grad_Uy (np.array) : gradient of the y-axis displacement field (LSQ method)
            
            Returns:
            - Bx (np.array) : source term in y-axis
        """  
        Bx = np.zeros((s.n_cells))

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells            
            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                weight = face['weight'] # Weight of the cell in the gradient calculation
                
                grad_Ux_ij = weight * grad_Ux[i] + (1-weight) * grad_Ux[j] # Weighted gradient of the x-axis displacement field
                grad_Uy_ij = weight * grad_Uy[i] + (1-weight) * grad_Uy[j] # Weighted gradient of the x-axis displacement field
                
                # x-axis oriented face
                Bx[i] += area * s.lambda_(xm, ym) * grad_Uy_ij[1] * normx
                # y-axis oriented face
                Bx[i] += area * s.mu(xm, ym) * grad_Uy_ij[0] * normy
                Bx[i] -= area * (s.mu(xm, ym)+s.lambda_(xm, ym)) * grad_Ux_ij[1] * normy
            
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                area = face['area'] # Length of the face
                
                # Boundary condition type on both axes
                cdt_type_x = s.b_cond['x'][face['bc_id']]['type']
                
                if cdt_type_x == 'displacement' or cdt_type_x == 'Displacement':                        
                    # x-axis oriented face
                    Bx[i] += area * normx * s.lambda_(xb, yb) * grad_Uy[i][1]
                    # y-axis oriented face
                    Bx[i] += area * normy * s.mu(xb, yb) * grad_Uy[i][0]
                    Bx[i] -= area * (s.mu(xb, yb)+s.lambda_(xb, yb)) * grad_Ux[i][1] * normy
        
        return Bx
    
    def source_transverse_y(s, grad_Ux : np.array = None, grad_Uy : np.array = None):
        """
            Calculate the source term for the transverse contribution for the y-axis component.
                        
            Parameters:
                - grad_Ux (np.array) : gradient of the x-axis displacement field (LSQ method)
            
            Returns:
            - By (np.array) : source term in y-axis
        """  
        By = np.zeros((s.n_cells))

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells            
            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                weight = face['weight'] # Weight of the cell in the gradient calculation

                grad_Ux_ij = weight * grad_Ux[i] + (1-weight) * grad_Ux[j] # Weighted gradient of the x-axis displacement field
                grad_Uy_ij = weight * grad_Uy[i] + (1-weight) * grad_Uy[j] # Weighted gradient of the y-axis displacement field
                    
                # y-axis oriented face
                By[i] += area * s.lambda_(xm, ym) * grad_Ux_ij[0] * normy
                # x-axis oriented face
                By[i] += area * s.mu(xm, ym) * grad_Ux_ij[1] * normx
                By[i] -= area * (s.mu(xm, ym)+s.lambda_(xm, ym)) * grad_Uy_ij[0] * normx
            
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                area = face['area'] # Length of the face
                
                # Boundary condition type on both axes
                cdt_type_y = s.b_cond['y'][face['bc_id']]['type']
                
                if cdt_type_y == 'displacement' or cdt_type_y == 'Displacement':                        
                    # y-axis oriented face
                    By[i] += area * normy * s.lambda_(xb, yb) * grad_Ux[i][0]
                    # x-axis oriented face
                    By[i] += area * normx * s.mu(xb, yb) * grad_Ux[i][1]
                    By[i] -= area * (s.mu(xb, yb)+s.lambda_(xb, yb)) * grad_Uy[i][0] * normx
        
        return By
  
    def source_boundary_x(s):
        """
            Calculate the source term for the boundary conditions for the x-axis component.
                        
            Returns:
            - Bx (np.array) : source term in y-axis
        """    
        Bx = np.zeros((s.n_cells))

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
                
                # x-axis boundary condition             
                if cdt_type_x == 'displacement' or cdt_type_x == 'Displacement':
                    Unt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Ux_b = Unt_b[0] * normx - Unt_b[1] * normy
                    # y-axis oriented face
                    Bx[i] += Ux_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx
                    # x-axis oriented face
                    Bx[i] += Ux_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy       
                elif cdt_type_x == 'stress' or cdt_type_x == 'Stress':
                    Tnt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Tx_b = Tnt_b[0] * normx - Tnt_b[1] * normy
                    Bx[i] += area * Tx_b
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        
        return Bx
  
    def source_boundary_y(s):
        """
            Calculate the source term for the boundary conditions for the y-axis component.
            
            Parameters:
                - By (np.array, default=None) : source term By. If None, it is initialized as a zero array.
            
            Returns:
            - By (np.array) : source term in y-axis
        """  
        By = np.zeros((s.n_cells))

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells                        
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                proj_distance = face['proj_distance'] # Distance between the centroid and the boundary point projected on the normal vector
                area = face['area'] # Length of the face
                
                # Boundary condition type on both axes
                cdt_type_y = s.b_cond['y'][face['bc_id']]['type']
                
                # y-axis boundary condition             
                if cdt_type_y == 'displacement' or cdt_type_y == 'Displacement':
                    Unt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Uy_b = Unt_b[0] * normy + Unt_b[1] * normx
                    # y-axis oriented face
                    By[i] += Uy_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normy * normy
                    # x-axis oriented face
                    By[i] += Uy_b * area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) / proj_distance * normx * normx        
                elif cdt_type_y == 'stress' or cdt_type_y == 'Stress':
                    Tnt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Ty_b = Tnt_b[0] * normy + Tnt_b[1] * normx
                    By[i] += area * Ty_b 
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        
        return By
   
    def source_correction_x(s, grad_Ux : np.array = None):
        """
            Calculate the source term for the grid correction (non-orthogonality + skewness) for the x-axis component.
                        
            Parameters:
                - grad_Ux (np.array) : gradient of the x-axis displacement field (LSQ method)
            
            Returns:
            - Bx (np.array) : source term in y-axis
        """  
        Bx = np.zeros((s.n_cells))

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

                grad_Ux_ij = weight * grad_Ux[i] + (1-weight) * grad_Ux[j] # Weighted gradient of the x-axis displacement field
                skew_correction = np.zeros(2)
                if distance[0]!=0:
                    skew_correction[0] = 1/distance[0] * (np.dot(skew, grad_Ux[j] - grad_Ux[i]))
                if distance[1]!=0:
                    skew_correction[1] = 1/distance[1] * (np.dot(skew, grad_Ux[j] - grad_Ux[i]))
                grad_Ux_ij += skew_correction 
                
                corr_grad_Ux_x = np.dot(grad_Ux_ij, orth/area*normx - tangential*normy)        
                corr_grad_Ux_y = np.dot(grad_Ux_ij, orth/area*normy + tangential*normx)          
                
                # x-axis oriented face
                Bx[i] += area * (2*s.mu(xm, ym) + s.lambda_(xm, ym)) * corr_grad_Ux_x * normx
                # y-axis oriented face
                Bx[i] += area * (2*s.mu(xm, ym) + s.lambda_(xm, ym)) * corr_grad_Ux_y * normy
            
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                area = face['area'] # Length of the face        
                orth = face['orth_correction'] # Skewness of the face
                tangential = face['tangential'] # Tangential vector to the face

                corr_grad_Ux_x = np.dot(grad_Ux[i], orth/area*normx - tangential*normy)        
                corr_grad_Ux_y = np.dot(grad_Ux[i], orth/area*normy + tangential*normx)        
                
                # x-axis component 
                cdt_type = s.b_cond['x'][face['bc_id']]['type']         
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    # x-axis oriented face
                    Bx[i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) * corr_grad_Ux_x * normx
                    # y-axis oriented face
                    Bx[i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) * corr_grad_Ux_y * normy      
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # No gradient if stress boundary condition which means no correction
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        
        return Bx
    
    def source_correction_y(s, grad_Uy : np.array = None):
        """
            Calculate the source term for the grid correction (non-orthogonality + skewness) for the y-axis component.
                        
            Parameters:
                - grad_Uy (np.array) : gradient of the y-axis displacement field (LSQ method)
                            
            Returns:
            - By (np.array) : source term in y-axis
        """  
        By = np.zeros((s.n_cells))

        for i, cell in enumerate(s.mesh.cells): # Iterate through cells            
            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid']
                xm, ym = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                weight = face['weight'] # Weight of the cell in the gradient calculation
                orth = face['orth_correction'] # Non-orthogonality correction
                skew = face['skew_correction'] # Skewness of the face
                distance = face['distance'] # Distance between the centroids
                tangential = face['tangential'] # Tangential vector to the face

                grad_Uy_ij = weight * grad_Uy[i] + (1-weight) * grad_Uy[j] # Weighted gradient of the x-axis displacement field
                skew_correction = np.zeros(2)
                if distance[0]!=0:
                    skew_correction[0] = 1/distance[0] * (np.dot(skew, grad_Uy[j] - grad_Uy[i]))
                if distance[1]!=0:
                    skew_correction[1] = 1/distance[1] * (np.dot(skew, grad_Uy[j] - grad_Uy[i]))
                grad_Uy_ij += skew_correction 
                
                corr_grad_Uy_x = np.dot(grad_Uy_ij, orth/area*normx - tangential*normy)        
                corr_grad_Uy_y = np.dot(grad_Uy_ij, orth/area*normy + tangential*normx)          
                
                # y-axis oriented face
                By[i] += area * (2*s.mu(xm, ym) + s.lambda_(xm, ym)) * corr_grad_Uy_y * normy
                # x-axis oriented face
                By[i] += area * (2*s.mu(xm, ym)+s.lambda_(xm, ym)) * corr_grad_Uy_x * normx
            
            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normal = face['normal'] # Normal vector to the face
                normx, normy = normal
                area = face['area'] # Length of the face        
                orth = face['orth_correction'] # Skewness of the face
                tangential = face['tangential'] # Tangential vector to the face

                corr_grad_Uy_x = np.dot(grad_Uy[i], orth/area*normx - tangential*normy)        
                corr_grad_Uy_y = np.dot(grad_Uy[i], orth/area*normy + tangential*normx)        
                
                # y-axis component 
                cdt_type = s.b_cond['y'][face['bc_id']]['type']         
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    # y-axis oriented face
                    By[i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) * corr_grad_Uy_y * normy
                    # x-axis oriented face
                    By[i] += area * (2*s.mu(xb, yb)+s.lambda_(xb, yb)) * corr_grad_Uy_x * normx      
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    pass # No gradient if stress boundary condition which means no correction
                else:
                    raise ValueError('Boundary condition type not recognized for y-axis')
        
        return By
    
    def solve(s, 
              max_iter : int, 
              tol_res : float = 0, 
              tol_res_norm : float = 0, 
              solver : Callable = scipy_sparse_spsolve, 
              precond : Callable = lambda *_ : None,
              precond_args : dict = dict(),
              early_stopping : bool = False,
              inc_trend_counter_max : int = 1,
              atol_scheduler : Callable = None):
        """
            Solve the elastic strain-stress problem for the given mesh and boundary conditions 
            using the Finite Volume Method and a segregated algorithm.\n
        
            Parameters:
                max_iter (int) : maximum number of iterations
                tol_res (float, default=0) : tolerance for the residual
                tol_res_norm (float, default=0) : tolerance for the normalized residual
                solver (Callable, default=inner_solver.scipy_sparse_spsolve) : function to solve the linear system of equations
                precond (Callable, default=None) : function to construct the preconditioner
                precond_args (dict, default={}) : arguments for the preconditioner
                early_stopping (bool, default=False) : flag to enable early stopping
                inc_trend_counter_max (int, default=1) : maximum number of increasing trends before stopping the iterations
                atol_scheduler (Callable, default=None) : function to update the tolerance for the inner solver
                
            Returns:
                Ux (np.array) x-axis displacement field    
                Uy (np.array) y-axis displacement field
        """
        ### Initialisation ###

        Ux, Uy = s.init_Ux.copy(), s.init_Uy.copy()
        
        # Early stopping increasing trend stack
        early_stopping_flag_x = False # Flag to stop the iterations for x-axis
        early_stopping_flag_y = False # Flag to stop the iterations for y-axis
        inc_trend_counter_x = 0
        inc_trend_counter_y = 0
        # If the increasing trend counter reaches inc_trend_counter_max in both directions
        # (early_stopping_flag_x and early_stopping_flag_y are True), the iterations are stopped
        
        # Construct the stiffness matrix and the body force source term
        Ax, Ay = s.stiffness()
        # Construct a preconditionner if provided
        Mx = precond(Ax, **precond_args) 
        My = precond(Ay, **precond_args)  
        
        Bx_f, By_f = s.source_body_force()
        Bx_b = s.source_boundary_x()
        By_b = s.source_boundary_y()
        
        # Construct the initial gradients and source terms
        grad_Ux, grad_Uy = s.grad(Ux), s.grad(Uy)
        Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
        By_t = s.source_transverse_y(grad_Ux, grad_Uy)
        Bx_c = s.source_correction_x(grad_Ux)
        By_c = s.source_correction_y(grad_Uy)
        Bx = lambda: Bx_t + Bx_b + Bx_f + Bx_c
        By = lambda: By_t + By_b + By_f + By_c
        
        # Store the initial statistics
        s.statistics.store(
            trend_x = 0, trend_y = 0,
            res_x = 0, res_y = 0,
            res_norm_x = 0, res_norm_y = 0,
            on_fly_res_x = 0, on_fly_res_y = 0,
            on_fly_res_norm_x = 0, on_fly_res_norm_y = 0,
            hist_Ux = Ux, hist_Uy = Uy,
            hist_Bx = Bx(),
            hist_By = By() 
        )
        
        inner_solver.ATOL = parameters.ATOL # Init the tolerance for the inner solver
        
        # Start the iterations
        for step in (pbar := tqdm(range(int(max_iter)))):
            outer_start_time = time()
            
            ## START OF OUTER ITERATION ##
            
            # Solve the system of equations for x-axis
            output = solver(Ax, Bx(), x0=Ux, M=Mx) # INNER ITERATIONS
            if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                Ux = output[0]
                inner_statistics_x = output[1]
            else:
                Ux = output
                inner_statistics_x = None
                        
            # Solve the system of equations for y-axis
            output = solver(Ay, By(), x0=Uy, M=My) # INNER ITERATIONS
            if  isinstance(output, tuple): # If the solver returns the solution and some statistics
                Uy = output[0]
                inner_statistics_y = output[1]
            else: 
                Uy = output
                inner_statistics_y = None
                
            grad_Ux = s.grad(Ux)
            grad_Uy = s.grad(Uy)
            # Update the source terms
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux)
            By_c = s.source_correction_y(grad_Uy)
            
            ## END OF OUTER ITERATION ##
            
            outer_end_time = time()
            
            # Store the statistics
            s.statistics.store(
                trend_x = np.linalg.norm(Ux-s.statistics.hist_Ux[-1],1),
                trend_y = np.linalg.norm(Uy-s.statistics.hist_Uy[-1],1),
                on_fly_res_x = residual(Ax, Ux, Bx()), on_fly_res_y = residual(Ay, Uy, By()),
                on_fly_res_norm_x = residual_norm(Ax, Ux, Bx()), on_fly_res_norm_y = residual_norm(Ay, Uy, By()),
                hist_Ux = Ux, hist_Uy = Uy,
                hist_Bx = Bx(),
                hist_By = By(),
                outer_iterations = {'time' : outer_end_time - outer_start_time},
                inner_iterations = {'x' : inner_statistics_x, 'y' : inner_statistics_y}
            )
            
            if atol_scheduler is not None:
                inner_solver.ATOL = atol_scheduler(s.statistics.on_fly_res_norm_x[-1],s.statistics.on_fly_res_norm_y[-1]) # Update the tolerance for the inner solver
                        
            # Print the progress of the iterations
            pbar.set_postfix_str(f"Normalized residuals: Rx {s.statistics.on_fly_res_norm_x[-1]:.2e}, Ry {s.statistics.on_fly_res_norm_y[-1]:.2e}, dRx {s.statistics.on_fly_res_norm_x[-1]-s.statistics.on_fly_res_norm_x[-2]:.2e}, dRy {s.statistics.on_fly_res_norm_y[-1]-s.statistics.on_fly_res_norm_y[-2]:.2e}")
    
            # Check early stopping criteria
            if step>0 and early_stopping:
                if not early_stopping_flag_x:
                    if s.statistics.trend_x[-1] >= s.statistics.trend_x[-2]: # Check if the trend is increasing
                        inc_trend_counter_x += 1 # Increment the counter
                    else:
                        inc_trend_counter_x = 0 # Reset the counter
                    if inc_trend_counter_x > inc_trend_counter_max: # Check if the counter is above the maximum
                        early_stopping_flag_x = True # Set the flag to stop the iterations
                if not early_stopping_flag_y:
                    if s.statistics.trend_y[-1] >= s.statistics.trend_y[-2]: # Check if the trend is increasing
                        inc_trend_counter_y += 1 # Increment the counter
                    else :
                        inc_trend_counter_y = 0 # Reset the counter
                    if inc_trend_counter_y > inc_trend_counter_max: # Check if the counter is above the maximum
                        early_stopping_flag_y = True # Set the flag to stop the iterations
                if early_stopping_flag_x and early_stopping_flag_y:
                    print('Early stopping after', step, 'iterations due to increasing trend')
                    break          
            # Convergence criterion on the fly (residuals)
            if s.statistics.on_fly_res_x[-1] < tol_res and s.statistics.on_fly_res_y[-1] < tol_res: # Check if the absolute residuals are below the tolerance
                print('Solution converged after', step, 'iterations based on the calculation of the residuals "on the fly"')
                break
            # Convergence criterion on the fly (normalized residuals)
            if s.statistics.on_fly_res_norm_x[-1] < tol_res_norm and s.statistics.on_fly_res_norm_y[-1] < tol_res_norm: 
                # Check if the normalized residuals are below the tolerance
                print('Solution converged after', step, 'iterations based on the calculation of the normalized residuals "on the fly"')
                break
        
        # Store the residuals using the converged source vector
        for i in range(step+2): # +1 (initial state), +1 (step starts at zero)
            s.statistics.store(
                res_x = residual(Ax, s.statistics.hist_Ux[i], Bx()),
                res_y = residual(Ay, s.statistics.hist_Uy[i], By()),
                res_norm_x = residual_norm(Ax, s.statistics.hist_Ux[i], Bx()),
                res_norm_y = residual_norm(Ay, s.statistics.hist_Uy[i], By())
            )
        
        # Store the extra statistics
        s.statistics.add('precond', precond.__name__)
        s.statistics.add('precond_args', precond_args)
        
        return Ux, Uy
    
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
        
        grad_Ux = s.grad(Ux)
        grad_Uy = s.grad(Uy)
        
        for i, cell in enumerate(s.mesh.cells):
            centroid = s.mesh.centroids[cell.centroid]
            Sxx[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Ux[i][0]
            Syy[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Uy[i][1]
            Sxy[i] = s.mu(centroid[0], centroid[1]) * (grad_Ux[i][1] + grad_Uy[i][0])
        
        return Sxx, Syy, Sxy
    
        
    def compute_stress_from_grad(s, grad_Ux : np.array, grad_Uy : np.array):
        """
            Compute the stress field on the mesh using the displacement field.\n
            Note that the finite volume methods here assumes a constant stress field in each cell.
            This approximation introduces an error in the stress field mainly at the boundaries of the domain.
            
            Parameters:
                - grad_Ux (np.array) : gradient of the x-axis displacement field
                - grad_Uy (np.array) : gradient of the y-axis displacement field
        """
        # Initialize the stress fields       
        Sxx = np.zeros((s.n_cells), dtype=DTYPE)
        Syy = np.zeros((s.n_cells), dtype=DTYPE)
        Sxy = np.zeros((s.n_cells), dtype=DTYPE)
        
        for i, cell in enumerate(s.mesh.cells):
            centroid = s.mesh.centroids[cell.centroid]
            Sxx[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Ux[i][0]
            Syy[i] = s.lambda_(centroid[0], centroid[1]) * (grad_Ux[i][0] + grad_Uy[i][1]) + 2 * s.mu(centroid[0], centroid[1]) * grad_Uy[i][1]
            Sxy[i] = s.mu(centroid[0], centroid[1]) * (grad_Ux[i][1] + grad_Uy[i][0])
        
        return Sxx, Syy, Sxy