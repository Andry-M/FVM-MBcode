# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 10/03/2025
# Context: INCEPTION – Investigating New Convergence schEmes for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Newton-Krylov extension of the Finite Volume Method (FVM) solver for MBcode
# -----------------------------------------------------------------------------

from mb_code.fvm import *
import numpy as np
from scipy.optimize import newton_krylov, BroydenFirst, KrylovJacobian, InverseJacobian


class StressStrain2d_Krylov(StressStrain2d):
    def __init__(s, mesh : Mesh2d, b_cond : dict, mu : Callable, lambda_ : Callable, f : Callable, alpha : float = 1, 
                 use_bgrad : bool = True, init_Ux : List = None, init_Uy : List = None):
        super().__init__(mesh=mesh,
                         b_cond=b_cond,
                         mu=mu, 
                         lambda_=lambda_, 
                         f=f,
                         alpha=alpha,
                         use_bgrad=use_bgrad,
                         init_Ux=init_Ux, 
                         init_Uy=init_Uy)
        
    # Residual using the Divergence of the stress tensor + Rhie-Chow stabilization term
    # r(U) = div(Sigma) + alpha * Kbar * (grad_U_imp - grad_U_exp)
    # Extracted from the solve method to be able to call it from outside the class.
    def div_stress_s4f(s, U):
        # Initialize the residual array
        residual = np.zeros(s.n_cells*2)
        
        # Split the displacement vector into x and y components
        Ux, Uy = U[:s.n_cells], U[s.n_cells:]
        
        # Compute the gradient of the displacement vector 
        grad_Ux, grad_Uy = s.grad(Ux, Uy)
        
        # Compute the stress tensor components
        Sxx, Syy, Sxy = s.compute_stress_from_grad(grad_Ux, grad_Uy)
        
        for i, cell in enumerate(s.mesh.cells): # Iterate through cells         

            # Iterate through inner faces
            for j, face in cell.stencil.items():
                fcentroid = face['fcentroid'] # Coordinates of the face centroid
                xf, yf = fcentroid # Coordinates of the face centroid
                normx, normy = face['normal'] # Normal vector to the face 
                area = face['area'] # Length of the face
                proj_distance = face['proj_distance'] # Distance between the centroids projected on the normal vector
                weight = face['weight'] # Weight of the cell in the gradient calculation
                distance = face['distance'] # Distance between the centroids
                
                Kbar = 2 * s.mu(xf, yf) + s.lambda_(xf, yf) # Laplacian modulus
                
                # Interpolate the stress tensor at the face center
                # Here using a naive interpolation method, but it can be improved
                # As I am using a structured mesh and the face is between two cells, I can use the average of the two cells
                Sxx_face = weight * Sxx[i] + (1-weight) * Sxx[j]
                Syy_face = weight * Syy[i] + (1-weight) * Syy[j]
                Sxy_face = weight * Sxy[i] + (1-weight) * Sxy[j]
                
                # Interpolate the gradient of the displacement vector at the face center
                grad_Ux_face = weight * grad_Ux[i] + (1-weight) * grad_Ux[j] # Weighted gradient of the x-axis displacement field
                grad_Uy_face = weight * grad_Uy[i] + (1-weight) * grad_Uy[j] # Weighted gradient of the x-axis displacement field
                
                # Compute the traction at the face center
                traction = [Sxx_face * normx + Sxy_face * normy, Sxy_face * normx + Syy_face * normy]
                
                # Add the Rhie-Chow term to the traction                
                imp_grad_x = (Ux[j] - Ux[i])/proj_distance
                imp_grad_y = (Uy[j] - Uy[i])/proj_distance
                
                traction[0] += s.alpha * Kbar * (imp_grad_x - grad_Ux_face[0] * distance[0]/proj_distance - grad_Ux_face[1] * distance[1]/proj_distance)
                traction[1] += s.alpha * Kbar * (imp_grad_y - grad_Uy_face[0] * distance[0]/proj_distance - grad_Uy_face[1] * distance[1]/proj_distance)
                
                traction = np.asarray(traction)*area
                
                residual[i] += traction[0]
                residual[i+s.n_cells] += traction[1]

            # Iterate through outer faces (boundaries)
            for b, face in cell.bstencil.items():
                xb, yb = s.mesh.bpoints[b] # Coordinates of the boundary point
                normx, normy = face['normal'] # Normal vector to the face
                proj_distance = face['proj_distance'] # Distance between the centroid and the boundary point projected on the normal vector
                area = face['area'] # Length of the face
                distance = face['distance'] # Distance between the centroid and the boundary point
                
                Kbar = 2 * s.mu(xb, yb) + s.lambda_(xb, yb) # Laplacian modulus
                
                traction = np.zeros(2) 
                
                # Boundary conditions are expressed in the (n, t) basis and have to be transformed to the (x, y) basis
                # x-axis component
                cdt_type = s.b_cond['x'][face['bc_id']]['type']
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    Unt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Ux_b = Unt_b[0] * normx - Unt_b[1] * normy
                    
                    # Compute the traction at the face center
                    traction[0] = Sxx[i] * normx + Sxy[i] * normy                        
                    # Add the Rhie-Chow term to the traction
                    imp_grad_x = (Ux_b - Ux[i])/proj_distance
                    
                    traction[0] += s.alpha * Kbar * (imp_grad_x - grad_Ux[i][0] * distance[0]/proj_distance - grad_Ux[i][1] * distance[1]/proj_distance)
                    
                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    Tnt_b = s.b_cond['x'][face['bc_id']]['value'](xb, yb)
                    Tx_b = Tnt_b[0] * normx - Tnt_b[1] * normy
                    traction[0] = Tx_b
                    
                # y-axis component
                cdt_type = s.b_cond['y'][face['bc_id']]['type']
                if cdt_type == 'displacement' or cdt_type == 'Displacement':
                    Unt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Uy_b = Unt_b[0] * normy + Unt_b[1] * normx

                    # Compute the traction at the face center
                    traction[1] = Sxy[i] * normx + Syy[i] * normy
                    # Add the Rhie-Chow term to the traction
                    imp_grad_y = (Uy_b - Uy[i])/proj_distance #+ corr_grad_Uy_x * distance[0]/proj_distance + corr_grad_Uy_y * distance[1]/proj_distance

                    traction[1] += s.alpha * Kbar * (imp_grad_y - grad_Uy[i][0] * distance[0]/proj_distance - grad_Uy[i][1] * distance[1]/proj_distance)

                elif cdt_type == 'stress' or cdt_type == 'Stress':
                    Tnt_b = s.b_cond['y'][face['bc_id']]['value'](xb, yb)
                    Ty_b = Tnt_b[0] * normy + Tnt_b[1] * normx
                    traction[1] = Ty_b
                
                traction = np.asarray(traction)*area
                
                residual[i] += traction[0] # x-axis component
                residual[i+s.n_cells] += traction[1] # y-axis component   

        return residual

    def solve(s,
              max_iter : int, 
              tol_res : float = 0, 
              tol_res_norm : float = 0, 
              solver : Callable = scipy_sparse_spsolve, 
              precond : Callable = lambda *_ : None,
              precond_args : dict = dict(),
              early_stopping : bool = False,
              inc_trend_counter_max : int = 1,
              before_nk_niter : int = 0,
              source_direct_update : bool = False,
              nk_residual : str = 'seg-fix-point',
              nk_res_rtol : float = 1e-10,
              nk_method : str = 'lgmres',
              nk_precond : Callable = lambda *_ : None,
              nk_precond_args : dict = dict()):
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
                before_nk_niter (int, default=0) : number of Picard iterations before the Newton-Krylov method
                source_direct_update (bool, default=False) : flag to enable direct update of the source term during Picard iteration
                nk_residual (str, default='fix-point') : type of residual for the Newton-Krylov method among ('fix-point', 'res-update', 'res-noupdate', 'res-norm-update', 'res-norm-noupdate', 'div-stress-s4f')
                nk_res_rtol (float, default=1e-10) : relative tolerance for the Newton-Krylov method (max-norm of the nk_residual)
                nk_method (str, default='lgmres') : method for the Newton-Krylov method among ('lgmres', 'gmres', 'bicgstab', 'cgs')
                nk_precond (Callable, default=lambda *_ : None) : function to construct the preconditioner for the Newton-Krylov method
                nk_precond_args (dict, default={}) : arguments for the preconditioner for the Newton-Krylov method  
            
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
        # (early_stopping_fl/ag_x and early_stopping_flag_y are True), the iterations are stopped
        
        # Construct the stiffness matrix and the body force source term
        Ax, Ay = s.stiffness()
        # Construct a preconditionner if provided
        Mx = precond(Ax, **precond_args)
        My = precond(Ay, **precond_args) 
        
        # Construct the constant parts of the source terms
        Bx_f, By_f = s.source_body_force()
        Bx_b = s.source_boundary_x()
        By_b = s.source_boundary_y()
        
        # Construct the initial gradients and source terms
        grad_Ux, grad_Uy = s.grad(Ux, Uy)
        Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
        By_t = s.source_transverse_y(grad_Ux, grad_Uy)
        Bx_c = s.source_correction_x(grad_Ux, Ux)
        By_c = s.source_correction_y(grad_Uy, Uy)

        # Store the initial statistics      
        s.statistics.store(
            trend_x = 0, trend_y = 0,
            res_x = 0, res_y = 0,
            res_norm_x = 0, res_norm_y = 0,
            on_fly_res_x = 0, on_fly_res_y = 0,
            on_fly_res_norm_x = 0, on_fly_res_norm_y = 0,
            hist_Ux = Ux, hist_Uy = Uy,
            hist_Bx = Bx_t + Bx_b + Bx_f + Bx_c,
            hist_By = By_t + By_b + By_f + By_c
        )  
            
        # Definition of the possible segregated iterations
        # Has to be methods to be called by the Newton-Krylov solver
        def segregated_iteration_direct_update(Ux, Uy, store=False):
            outer_start_time = time()
            
            grad_Ux, grad_Uy = s.grad(Ux, Uy)
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux, Ux)
            
            # Solve the system of equations for x-axis
            output = solver(Ax, Bx_t + Bx_b + Bx_f + Bx_c, x0=Ux, M=Mx) # INNER ITERATIONS
            if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                Ux = output[0]
                inner_statistics_x = output[1]
            else:
                Ux = output
                inner_statistics_x = None

            grad_Ux, grad_Uy = s.grad(Ux, Uy)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            By_c = s.source_correction_y(grad_Uy, Uy)
                        
            # Solve the system of equations for y-axis
            output = solver(Ay, By_t + By_b + By_f + By_c, x0=Uy, M=My) # INNER ITERATIONS
            if  isinstance(output, tuple): # If the solver returns the solution and some statistics
                Uy = output[0]
                inner_statistics_y = output[1]
            else: 
                Uy = output
                inner_statistics_y = None
            
            outer_end_time = time()
            
            if store:
                grad_Ux, grad_Uy = s.grad(Ux, Uy)
                Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
                By_t = s.source_transverse_y(grad_Ux, grad_Uy)
                Bx_c = s.source_correction_x(grad_Ux, Ux)
                By_c = s.source_correction_y(grad_Uy, Uy)
                s.statistics.store(
                    trend_x = np.linalg.norm(Ux-s.statistics.hist_Ux[-1],1),
                    trend_y = np.linalg.norm(Uy-s.statistics.hist_Uy[-1],1),
                    on_fly_res_x = residual(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_y = residual(Ay, Uy, By_t + By_b + By_f + By_c),
                    on_fly_res_norm_x = residual_norm(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_norm_y = residual_norm(Ay, Uy, By_t + By_b + By_f + By_c),
                    hist_Ux = Ux, hist_Uy = Uy,
                    hist_Bx = Bx_t + Bx_b + Bx_f + Bx_c,
                    hist_By = By_t + By_b + By_f + By_c,
                    outer_iterations = {'time' : outer_end_time - outer_start_time},
                    inner_iterations = {'x' : inner_statistics_x, 'y' : inner_statistics_y}
                )
            
            return Ux, Uy
        
        def segregated_iteration(Ux, Uy, store=False):
            outer_start_time = time()
            
            grad_Ux, grad_Uy = s.grad(Ux, Uy)
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux, Ux)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            By_c = s.source_correction_y(grad_Uy, Uy)
            
            # Solve the system of equations for x-axis
            output = solver(Ax, Bx_t + Bx_b + Bx_f + Bx_c, x0=Ux, M=Mx) # INNER ITERATIONS
            if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                Ux = output[0]
                inner_statistics_x = output[1]
            else:
                Ux = output
                inner_statistics_x = None
            
            # Solve the system of equations for y-axis
            output = solver(Ay, By_t + By_b + By_f + By_c, x0=Uy, M=My) # INNER ITERATIONS
            if  isinstance(output, tuple): # If the solver returns the solution and some statistics
                Uy = output[0]
                inner_statistics_y = output[1]
            else: 
                Uy = output
                inner_statistics_y = None
            
            outer_end_time = time()
            
            if store:
                grad_Ux, grad_Uy = s.grad(Ux, Uy)
                Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
                Bx_c = s.source_correction_x(grad_Ux, Ux)
                By_t = s.source_transverse_y(grad_Ux, grad_Uy)
                By_c = s.source_correction_y(grad_Uy, Uy)
                s.statistics.store(
                        trend_x = np.linalg.norm(Ux-s.statistics.hist_Ux[-1],1),
                        trend_y = np.linalg.norm(Uy-s.statistics.hist_Uy[-1],1),
                        on_fly_res_x = residual(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_y = residual(Ay, Uy, By_t + By_b + By_f + By_c),
                        on_fly_res_norm_x = residual_norm(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_norm_y = residual_norm(Ay, Uy, By_t + By_b + By_f + By_c),
                        hist_Ux = Ux, hist_Uy = Uy,
                        hist_Bx = Bx_t + Bx_b + Bx_f + Bx_c,
                        hist_By = By_t + By_b + By_f + By_c,
                        outer_iterations = {'time' : outer_end_time - outer_start_time},
                        inner_iterations = {'x' : inner_statistics_x, 'y' : inner_statistics_y}
                )
            
            return Ux, Uy
        
        # Definition of the different Newton-Krylov residuals
        # Fixed-point residual r(U) = S(U) - U
        def seg_fix_point(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            seg_iter = iteration(Ux, Uy)
            return diff_map(np.concatenate(seg_iter), U)
        
        # Fixed point residual using source calculated directly from A:
        # r(U) = B(k+1) - B(k) --> A(U(k+1)) - A(U(k)) = A(U(k+1) - U(k))
        def seg_fix_point_source_a(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            seg_iter = iteration(Ux, Uy)
            return np.array(A @ (np.concatenate(seg_iter) - U)).flatten()
            
        # Fixed point residual using source calculated from the gradients:
        # r(U) = B(U(k+1)) - B(U(k)) = B(k+2) - B(k+1)
        def seg_fix_point_source_b(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            
            old_grad_Ux, old_grad_Uy = s.grad(Ux, Uy)
            old_Bx_t = s.source_transverse_x(old_grad_Ux, old_grad_Uy)
            old_By_t = s.source_transverse_y(old_grad_Ux, old_grad_Uy)
            old_Bx_c = s.source_correction_x(Ux, old_grad_Ux)
            old_By_c = s.source_correction_y(Uy, old_grad_Uy)

            seg_iter = iteration(Ux, Uy)
            
            new_grad_Ux, new_grad_Uy = s.grad(seg_iter[0], seg_iter[1])
            new_Bx_t = s.source_transverse_x(new_grad_Ux, new_grad_Uy)
            new_By_t = s.source_transverse_y(new_grad_Ux, new_grad_Uy)
            new_Bx_c = s.source_correction_x(seg_iter[0], new_grad_Ux)
            new_By_c = s.source_correction_y(seg_iter[1], new_grad_Uy)
                        
            return diff_map(np.concatenate((new_Bx_t+new_Bx_c, new_By_t+new_By_c)), np.concatenate((old_Bx_t+old_Bx_c, old_By_t+old_By_c)))
        
        # Fixed point residual using the segregated algorithm residual
        # r(U) = (AU(k+1) - B(U(k+1))) - (AU(k) - B(U(k)))
        def seg_fix_point_res(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            
            # Construct the initial gradients and source terms
            grad_Ux, grad_Uy = s.grad(Ux, Uy)
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux, Ux)
            By_c = s.source_correction_y(grad_Uy, Uy)
            
            old_res_x = residual_map(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c)
            old_res_y = residual_map(Ay, Uy, By_t + By_b + By_f + By_c)
            
            seg_iter = iteration(Ux, Uy)
            
            # Construct the initial gradients and source terms
            grad_Ux, grad_Uy = s.grad(seg_iter[0], seg_iter[1])
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(seg_iter[0], grad_Ux)
            By_c = s.source_correction_y(seg_iter[1], grad_Uy)
            
            res_x = residual_map(Ax, seg_iter[0], Bx_t + Bx_b + Bx_f + Bx_c)
            res_y = residual_map(Ay, seg_iter[1], By_t + By_b + By_f + By_c)
            
            return np.concatenate([res_x-old_res_x, res_y-old_res_y])
        
        # Residual using the segregated algorithm residual
        # r(U) = AU(k) - B(U(k))
        def seg_res(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            
            # Construct the initial gradients and source terms
            grad_Ux, grad_Uy = s.grad(Ux, Uy)
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux, Ux)
            By_c = s.source_correction_y(grad_Uy, Uy)
            
            res_x = residual_map(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c)
            res_y = residual_map(Ay, Uy, By_t + By_b + By_f + By_c)
            
            return np.concatenate([res_x, res_y])
        
        # Residual using the segregated algorithm residual after an iteration
        # r(U) = AU(k+1) - B(U(k+1))
        def seg_res_update(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            
            seg_iter = iteration(Ux, Uy)
            
            # Construct the initial gradients and source terms
            grad_Ux, grad_Uy = s.grad(seg_iter[0], seg_iter[1])
            Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
            By_t = s.source_transverse_y(grad_Ux, grad_Uy)
            Bx_c = s.source_correction_x(grad_Ux, Ux)
            By_c = s.source_correction_y(grad_Uy, Uy)
            
            res_x = residual_map(Ax, seg_iter[0], Bx_t + Bx_b + Bx_f + Bx_c)
            res_y = residual_map(Ay, seg_iter[1], By_t + By_b + By_f + By_c)
            
            return np.concatenate([res_x, res_y])
        
        # Residual using the Divergence of the stress tensor + Rhie-Chow stabilization term after an iteration
        # r(U) = div(Sigma) + alpha * Kbar * (grad_U_imp - grad_U_exp)
        def div_stress_s4f_update(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            seg_iter = iteration(Ux, Uy)
            return s.div_stress_s4f(np.concatenate(seg_iter))

        # Fixed point residual using the Divergence of the stress tensor + Rhie-Chow stabilization term after an iteration
        # r(U) = [div(Sigma) + alpha * Kbar * (grad_U_imp - grad_U_exp)](k+1) - [div(Sigma) + alpha * Kbar * (grad_U_imp - grad_U_exp)](k)
        # Force to use a different alpha for the two iterations to avoid the same value
        def div_stress_s4f_diff(U):
            Ux, Uy = U[:s.n_cells], U[s.n_cells:]
            
            seg_iter = iteration(Ux, Uy)
            
            old_div_stress_s4f = s.div_stress_s4f(U)
            new_div_stress_s4f = s.div_stress_s4f(np.concatenate([seg_iter[0], seg_iter[1]]))
            return new_div_stress_s4f - old_div_stress_s4f
        
        
        # Chose the selected iteration algorithm
        if source_direct_update:
            iteration = segregated_iteration_direct_update
        else:
            iteration = segregated_iteration
                 
        # Chose the selected residual
        if nk_residual == 'seg-fix-point':
            krylov_residual = seg_fix_point
        elif nk_residual == 'seg-fix-point-source-a':
            krylov_residual = seg_fix_point_source_a
        elif nk_residual == 'seg-fix-point-source-b':
            krylov_residual = seg_fix_point_source_b
        elif nk_residual == 'seg-fix-point-res':
            krylov_residual = seg_fix_point_res
        elif nk_residual == 'seg-res':
            krylov_residual = seg_res
        elif nk_residual == 'seg-res-update':
            krylov_residual = seg_res_update
        elif nk_residual == 'div-stress-s4f':
            krylov_residual = s.div_stress_s4f
        elif nk_residual == 'div-stress-s4f-update':
            krylov_residual = div_stress_s4f_update
        elif nk_residual == 'div-stress-s4f-diff':
            krylov_residual = div_stress_s4f_diff
        else:
            raise ValueError(f"Unknown residual type: {nk_residual}. Available types are: seg-fix-point, seg-fix-point-source-a, seg-fix-point-source-b, seg-fix-point-res, seg-res, seg-res-update, div-stress-s4f, div-stress-s4f-update, div-stress-s4f-diff")
        
        # Run the Picard iterations before the Newton-Krylov method
        if before_nk_niter > 0:
            print('Starting Picard iterations before Newton-Krylov')
            for step in (pbar := tqdm(range(int(before_nk_niter)))):
                Ux, Uy = iteration(Ux, Uy, store=True)
                pbar.set_postfix_str(f"Normalized residuals: Rx {s.statistics.on_fly_res_norm_x[-1]:.2e}, Ry {s.statistics.on_fly_res_norm_y[-1]:.2e}, dRx {s.statistics.on_fly_res_norm_x[-1]-s.statistics.on_fly_res_norm_x[-2]:.2e}, dRy {s.statistics.on_fly_res_norm_y[-1]-s.statistics.on_fly_res_norm_y[-2]:.2e}")
            print('Picard iterations before Newton-Krylov completed\n')
            
        # Define the callback method of Newton-Krylov to store the convergence
        s.statistics.add('nk_Ux', [])
        s.statistics.add('nk_Uy', [])
        s.statistics.add('nk_time', [])
        s.statistics.add('nk_GMRes_size', [])
        s.statistics.add('nk_div_stress_s4f', [])
        s.statistics.add('nk_residual', [])
        
        n = 0
                
        def inner_callback(pr_norm):
            nonlocal n
            n += 1
        
        def callback(x, f):
            nonlocal n
            s.statistics.store(
                nk_time = time(),
                nk_Ux = x[:s.n_cells],
                nk_Uy = x[s.n_cells:],
                nk_GMRes_size = n,
                nk_div_stress_s4f = s.div_stress_s4f(x),
                nk_residual = f,
            )
            n = 0
            print('Callback called, div_stress_s4f:', np.linalg.norm(s.statistics.nk_div_stress_s4f[-1]))
        
        # Construct the segregated algorithm global matrix for the preconditioner
        A = np.zeros((s.n_cells * 2, s.n_cells * 2))
        A[:s.n_cells, :s.n_cells] = Ax.toarray()
        A[s.n_cells:, s.n_cells:] = Ay.toarray()
    
        print('Starting Newton-Krylov iterations')
        newton_start_time = time()

        # Use the same preconditionner as for the segregated algorithm
        M = nk_precond(sps.csc_matrix(A), **nk_precond_args) 
        callback(np.concatenate([Ux, Uy]), np.zeros(s.n_cells*2))
        print('Initial residual:', np.linalg.norm(krylov_residual(np.concatenate([Ux, Uy]))))
        
        U_newton = newton_krylov(krylov_residual, 
                                np.concatenate([Ux, Uy]), 
                                verbose=True,
                                method = nk_method, 
                                inner_M=M, 
                                f_rtol=nk_res_rtol,
                                f_tol=1e-3, # Safer compared to the default 6e-6
                                callback=callback, 
                                inner_callback=inner_callback,
                                tol_norm=np.linalg.norm) # Force the use of l2 norm 
                                #inner_inner_m = 50,
                                #inner_restart=50, # Large restart to get the size of the Krylov subspace
                                #inner_callback_type='pr_norm',
                                #inner_rtol = 1e-10, 
                                #iter = 35)
                                #maxiter = 30)
        
        s.statistics.store(
            nk_time = time(),
        )
        Ux, Uy = U_newton[:s.n_cells], U_newton[s.n_cells:]
        grad_Ux, grad_Uy = s.grad(Ux, Uy)
        Bx_t = s.source_transverse_x(grad_Ux, grad_Uy)
        By_t = s.source_transverse_y(grad_Ux, grad_Uy)
        Bx_c = s.source_correction_x(grad_Ux, Ux)
        By_c = s.source_correction_y(grad_Uy, Uy)
        newton_end_time = time()
        s.statistics.store(
                trend_x = np.linalg.norm(Ux-s.statistics.hist_Ux[-1],1),
                trend_y = np.linalg.norm(Uy-s.statistics.hist_Uy[-1],1),
                on_fly_res_x = residual(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_y = residual(Ay, Uy, By_t + By_b + By_f + By_c),
                on_fly_res_norm_x = residual_norm(Ax, Ux, Bx_t + Bx_b + Bx_f + Bx_c), on_fly_res_norm_y = residual_norm(Ay, Uy, By_t + By_b + By_f + By_c),
                hist_Ux = Ux, hist_Uy = Uy,
                hist_Bx = Bx_t + Bx_b + Bx_f + Bx_c,
                hist_By = By_t + By_b + By_f + By_c,
                outer_iterations = {'time' : newton_end_time - newton_start_time},
                inner_iterations = {'x' : 'Newton', 'y' : "Newton"}
            )
        print('Newton-Krylov iterations completed')
        
        # Start the iterations
        for step in (pbar := tqdm(range(int(max_iter)))):
            
            Ux, Uy = iteration(Ux, Uy, store=True)

            # Print the progress of the iterations
            pbar.set_postfix_str(f"Normalized residuals: Rx {s.statistics.on_fly_res_norm_x[-1]:.2e}, Ry {s.statistics.on_fly_res_norm_y[-1]:.2e}, dRx {s.statistics.on_fly_res_norm_x[-1]-s.statistics.on_fly_res_norm_x[-2]:.2e}, dRy {s.statistics.on_fly_res_norm_y[-1]-s.statistics.on_fly_res_norm_y[-2]:.2e}")
    
            # Check early stopping criteria
            if step>0:
                if early_stopping:
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
        
        if max_iter > 0:
            # Store the residuals using the converged source vector
            for i in range(1+before_nk_niter+step+1+1): # +1 (initial state), +1 (step starts at zero), +1 (Newton-Krylov iterations)
                s.statistics.store(
                    res_x = residual(Ax, s.statistics.hist_Ux[i], s.statistics.hist_Bx[-1]),
                    res_y = residual(Ay, s.statistics.hist_Uy[i], s.statistics.hist_By[-1]),
                    res_norm_x = residual_norm(Ax, s.statistics.hist_Ux[i], s.statistics.hist_Bx[-1]),
                    res_norm_y = residual_norm(Ay, s.statistics.hist_Uy[i], s.statistics.hist_By[-1])
                )
        
        # Store the extra statistics
        s.statistics.add('precond', precond.__name__)
        s.statistics.add('precond_args', precond_args)

        return Ux, Uy