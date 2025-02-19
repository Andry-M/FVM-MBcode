# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 28/11/2024
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Machine Learning extension of the Finite Volume Method (FVM) solver for MBcode
# -----------------------------------------------------------------------------

from mb_code.fvm import *

class StressStrain2d_Anderson(StressStrain2d):
    def __init__(s, mesh : Mesh2d, b_cond : dict, mu : Callable, lambda_ : Callable, f : Callable,
                 init_Ux : List = None, init_Uy : List = None):
        super().__init__(mesh=mesh,
                         b_cond=b_cond,
                         mu=mu, 
                         lambda_=lambda_, 
                         f=f, 
                         init_Ux=init_Ux, 
                         init_Uy=init_Uy)

    def solve(s, 
              max_iter : int, 
              tol_res : float = 0, 
              tol_res_norm : float = 0, 
              solver : Callable = scipy_sparse_spsolve, 
              precond : Callable = lambda *_ : None,
              precond_args : dict = dict(),
              early_stopping : bool = False,
              inc_trend_counter_max : int = 1,
              anderson_order : int = 30,
              alpha : float = 1.0):
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
                anderson_order (int, default=30) : number of previous solutions to use in the Anderson mixing algorithm
                
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
        Bx_t = s.source_transverse_x(grad_Uy)
        By_t = s.source_transverse_y(grad_Ux)
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
            hist_Bx = {'transverse' : Bx_t, 'boundary' : Bx_b, 'force' : Bx_f, 'correction' : Bx_c, 'all' : Bx()},
            hist_By = {'transverse' : By_t, 'boundary' : By_b, 'force' : By_f, 'correction' : By_c, 'all' : By()} 
        )
        
        # Start the iterations
        for step in (pbar := tqdm(range(int(max_iter)))):
            outer_start_time = time()
            
            ## START OF OUTER ITERATION ##
            lag = 2 # Lag for the Anderson mixing algorithm to avoid using transient solutions
            if step>2 and (step-lag) % anderson_order == 0: # If the order is reached, use the Anderson mixing algorithm
                print(f'Step {step} - Anderson mixing algorithm applied')
                # Compute the error vectors
                error_x = np.zeros((anderson_order, s.n_cells))
                error_y = np.zeros((anderson_order, s.n_cells))
                for o in range(anderson_order):
                    ind = step-anderson_order+o
                    error_x[o,:] = s.statistics.hist_Ux[ind+1]-s.statistics.hist_Ux[ind]
                    error_y[o,:] = s.statistics.hist_Uy[ind+1]-s.statistics.hist_Uy[ind]
                # Compute the projection matrix
                T_x = np.zeros((anderson_order, anderson_order))
                T_y = np.zeros((anderson_order, anderson_order))
                for i in range(anderson_order):
                    for j in range(i, anderson_order):
                        T_x[i,j] = np.dot(error_x[i], error_x[j])
                        T_y[i,j] = np.dot(error_y[i], error_y[j])
                # Normalize the projection matrix
                T_x = T_x / np.max(T_x)
                T_y = T_y / np.max(T_y)
                for i in range(anderson_order):
                    for j in range(i):
                        T_x[i,j] = T_x[j,i]
                        T_y[i,j] = T_y[j,i]
                # Compute the coefficients
                anderson_coeffs_x = np.linalg.solve(T_x, np.ones(anderson_order))
                anderson_coeffs_x /= np.sum(anderson_coeffs_x)
                anderson_coeffs_y = np.linalg.solve(T_y, np.ones(anderson_order))
                anderson_coeffs_y /= np.sum(anderson_coeffs_y)
                
                # Compute the new displacement components
                Ux = np.zeros(s.n_cells)
                Uy = np.zeros(s.n_cells)
                for o in range(anderson_order):
                    Ux += anderson_coeffs_y[o]*s.statistics.hist_Ux[step-anderson_order+o]
                    Uy += anderson_coeffs_x[o]*s.statistics.hist_Uy[step-anderson_order+o]
                
                Ux = alpha*Ux + (1-alpha)*s.statistics.hist_Ux[-1]
                Uy = alpha*Uy + (1-alpha)*s.statistics.hist_Uy[-1]
                
                # Update the source terms
                grad_Ux = s.grad(Ux)
                grad_Uy = s.grad(Uy)
                Bx_c = s.source_correction_x(grad_Ux, Bx_c)
                By_c = s.source_correction_y(grad_Uy, By_c)
                Bx_t = s.source_transverse_x(grad_Uy, Bx_t)
                #By_t = s.source_transverse_y(grad_Ux, By_t) # Re-updated later, before use
            
            else:
                # Solve the system of equations for x-axis
                output = solver(Ax, Bx(), x0=Ux, M=Mx) # INNER ITERATIONS
                if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                    Ux = output[0]
                    inner_statistics_x = output[1]
                else:
                    Ux = output
                    inner_statistics_x = None
                grad_Ux = s.grad(Ux)
                # Update the source terms
                By_t = s.source_transverse_y(grad_Ux, By_t)
                Bx_c = s.source_correction_x(grad_Ux, Bx_c)
                            
                # Solve the system of equations for y-axis
                output = solver(Ay, By(), x0=Uy, M=My) # INNER ITERATIONS
                if  isinstance(output, tuple): # If the solver returns the solution and some statistics
                    Uy = output[0]
                    inner_statistics_y = output[1]
                else: 
                    Uy = output
                    inner_statistics_y = None
                grad_Uy = s.grad(Uy)
                # Update the source terms
                Bx_t = s.source_transverse_x(grad_Uy, Bx_t)
                By_c = s.source_correction_y(grad_Uy, By_c)
            
            ## END OF OUTER ITERATION ##
            
            outer_end_time = time()
            
            # Store the statistics
            s.statistics.store(
                trend_x = np.linalg.norm(Ux-s.statistics.hist_Ux[-1],1),
                trend_y = np.linalg.norm(Uy-s.statistics.hist_Uy[-1],1),
                on_fly_res_x = residual(Ax, Ux, Bx()), on_fly_res_y = residual(Ay, Uy, By()),
                on_fly_res_norm_x = residual_norm(Ax, Ux, Bx()), on_fly_res_norm_y = residual_norm(Ay, Uy, By()),
                hist_Ux = Ux, hist_Uy = Uy,
                hist_Bx = {'transverse' : Bx_t, 'boundary' : Bx_b, 'force' : Bx_f, 'correction' : Bx_c, 'all' : Bx()},
                hist_By = {'transverse' : By_t, 'boundary' : By_b, 'force' : By_f, 'correction' : By_c, 'all' : By()},
                outer_iterations = {'time' : outer_end_time - outer_start_time},
                inner_iterations = {'x' : inner_statistics_x, 'y' : inner_statistics_y}
            )
            
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