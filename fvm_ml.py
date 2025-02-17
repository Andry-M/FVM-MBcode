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

import torch

class StressStrain2dML(StressStrain2d):
    def __init__(s, mesh : Mesh2d, b_cond : dict, mu : Callable, lambda_ : Callable, f : Callable):
        super().__init__(mesh, b_cond, mu, lambda_, f)
        
    def solve(s, max_iter : int, 
              tol_res : float = 0, 
              tol_res_norm : float = 0,
              tol_trend : float = 0, 
              continue_ : bool = False,
              statistics : dict = None, 
              solver : Callable = scipy_sparse_spsolve,
              precond : Callable = lambda *_ : None,
              precond_args : dict = dict(),
              early_stopping : bool = False,
              inc_trend_counter_max : int = 1, ML_model : Callable = None, 
              scaler_features = None, scaler_targets = None):
        """
            Solve the elastic stress problem for the given mesh and boundary conditions using the Finite Volume Method
            and a segregated algorithm.\n
            Uses a Machine Learning model to fast-forward the iterations.\n
            The solver is based on a scipy sparse solver function.\n
            
            The returned statistics dictionary contains the following data
            - 'trend' ({'x' : List[float], 'y' : List[float]}) : trends of the displacement fields
            - 'res' ({'x' : List[float], 'y' : List[float]}) : absolute residuals
            - 'norm_res' ({'x' : List[float], 'y' : List[float]}) : normalized residuals (OpenFOAM-like)
            - 'on_fly_res' ({'x' : List[float], 'y' : List[float]}) : absolute residuals between iterations
            - 'on_fly_res_norm' ({'x' : List[float], 'y' : List[float]}) : normalized residuals between iterations
            - 'hist_Ux' (List[np.array]): the history of the x-axis displacement field
            - 'hist_Uy' (List[np.array]): the history of the y-axis displacement field
            - 'hist_Bx' (List[dict]): the history of the x-axis source vectors
            - 'hist_By' (List[dict]): the history of the y-axis source vectors
            - 'outer_iteration_times' (List[float]): the time taken for each outer iteration
            - 'inner_statistics' ({'x' : ?, 'y' : ?}): the statistics of the inner iterations
                
            Parameters:
                - max_iter (int) : the maximum number of outer iterations
                - tol_res (float, default=0) : the tolerance on the residuals for convergence stopping
                - tol_res_norm (float, default=0) : the tolerance on the normalized residuals for convergence stopping
                - tol_trend (float, default=0) : the tolerance on the trend of the displacement fields for convergence stopping
                - continue\_ (bool, default=False) : if True, the solution is continued from the previous one
                - statistics (dict, default=None) : the dictionary to store the statistics of the solution
                - solver (Callable, default=scipy.sparse.linalg.spsolve) : the scipy sparse solver function to use to solve the system of equations, should return only the solution
                - precond (Callable, default= lambda *_ : None) : the preconditionner to use for the solver
                - precond_args (dict, default=dict()) : the arguments to pass to the preconditionner
                - early_stopping (bool, default=False) : if True, the early stopping is activated based on the trend of the displacement fields
                - inc_trend_counter_max (int, default=1) : the maximum number of increasing trend to stop the iterations
                - ML_model (Callable, default=None) : the Machine Learning model to use to predict the displacement fields
                - scaler_features (default=None) : the scaler to use for the input of the Machine Learning model
                - scaler_targets (default=None) : the scaler to use for the output of the Machine Learning model
                
            Returns:
            - Ux (np.array) : x-axis displacement field               
            - Uy (np.array) : y-axis displacement field
            - statistics (dict) : the dictionary storing the statistics of the solution
        """
        ### Start of statistics ###
        
        # Check if the solution is continued that the right statistics are provided
        if continue_:
            required_keys = ['trend', 'hist_Ux', 'hist_Uy', 'hist_Bx', 'hist_By', 'outer_iteration_times',
                             'on_fly_res', 'on_fly_res_norm', 'inner_statistics']
            if statistics is None:
                raise ValueError('The statistics dictionary must be provided when continuing the solution')
            if not all(key in statistics.keys() for key in required_keys):
                raise ValueError(f'Some required keys {required_keys} are missing in the statistics dictionary')
            for key in required_keys:
                if statistics[key] is None:
                    raise ValueError(f'The key {key} in the statistics dictionary is None \
                        and must be provided when continuing the solution')
            # Get the last statistics to append the new ones
            Ux, Uy = statistics['hist_Ux'][-1], statistics['hist_Uy'][-1]
            hist_Ux, hist_Uy = statistics['hist_Ux'], statistics['hist_Uy']
            hist_Bx, hist_By = statistics['hist_Bx'][:-1], statistics['hist_By'][:-1]
            trend = statistics['trend']
            outer_times = statistics['outer_iteration_times']
            inner_statistics_x = statistics['inner_statistics']['x'] # Array of 'inner statistics' of any kind for x-axis
            inner_statistics_y = statistics['inner_statistics']['y'] # Array of 'inner statistics' of any kind for y-axis
            on_fly_res = statistics['on_fly_res']
            on_fly_res_norm = statistics['on_fly_res_norm']
        else: 
            # Initialize the displacement field
            Ux, Uy = np.zeros(s.n_cells), np.zeros(s.n_cells)
            # Dictionary to store all the statistics
            statistics = {}
            # Trend of the displacement fields
            trend = {'x' : [], 'y' : []}
            # History of the displacement fields
            hist_Ux = [Ux] 
            hist_Uy = [Uy]
            # History of the source vectors
            hist_Bx = []
            hist_By = []
            # List of times per outer iteration
            outer_times = []
            # Information to store "on the fly" for early stopping
            # 0 values are used for the first iteration progression bar display
            on_fly_res = {'x' : [0], 'y' : [0]}
            on_fly_res_norm = {'x' : [0], 'y' : [0]}
            # List of inner statistics
            inner_statistics_x = []
            inner_statistics_y = []
            
        # Final residuals computed using the converged source vector
        res = {'x' : [], 'y' : []}
        res_norm = {'x' : [], 'y' : []}
        
        ### End of statistics ###
        
        # Early stopping increasing trend stack
        early_stopping_flag_x = False # Flag to stop the iterations for x-axis
        early_stopping_flag_y = False # Flag to stop the iterations for y-axis
        inc_trend_counter_x = 0
        inc_trend_counter_y = 0
        # If the increasing trend counter reaches inc_trend_counter_max in both directions
        # (early_stopping_flag_x and early_stopping_flag_y are True), the iterations are stopped
        
        # Check number of consecutive steps to calculate before using the neural network
        if ML_model is not None:
            input_dim = ML_model.input_dim
            nb_iteration_per_input = int(input_dim//2)
                    
        ### Start of solver ###
        
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
        
        # Store the initial source vectors
        hist_Bx.append({'transverse' : Bx_t, 'boundary' : Bx_b, 'force' : Bx_f, 'correction' : Bx_c, 'all' : Bx()})
        hist_By.append({'transverse' : By_t, 'boundary' : By_b, 'force' : By_f, 'correction' : By_c, 'all' : By()})
        
        for step in (pbar := tqdm(range(1,int(max_iter+1)))):
            outer_start_time = time()
            
            ## START OF OUTER ITERATION ##
            
            if ML_model is not None and step == nb_iteration_per_input:
                ML_Ux, ML_Uy = np.zeros_like(Ux), np.zeros_like(Uy)
                for cell in range(s.n_cells):
                    ML_input = []
                    for i in range(nb_iteration_per_input):
                        ML_input.append(hist_Ux[i][cell])
                        ML_input.append(hist_Uy[i][cell])
                    ML_input = torch.tensor(ML_input, dtype=torch.float32).reshape(1,-1)
                    ML_input = scaler_features.transform(ML_input)
                    
                    ML_output = ML_model(ML_input)
                    ML_output = scaler_targets.inverseTransform(ML_output.cpu()).detach().numpy()[0]
                    ML_Ux[cell] = ML_output[0]
                    ML_Uy[cell] = ML_output[1]
                
                Ux = np.array(ML_Ux)
                Uy = np.array(ML_Uy)
                grad_Ux = s.grad(Ux)
                grad_Uy = s.grad(Uy)
                Bx_t = s.source_transverse_x(grad_Uy, Bx_t)
                Bx_c = s.source_correction_x(grad_Ux, Bx_c) # will be reupdated later before use but calculated for storage
                By_t = s.source_transverse_y(grad_Ux, By_t) # will be reupdated later before use but calculated for storage
                By_c = s.source_correction_y(grad_Uy, By_c)
                inner_statistics_x.append({'info' : 'ml', 'iterations' : None})
                inner_statistics_y.append({'info' : 'ml', 'iterations' : None})
            
            else:
                   
                # Solve the system of equations for x-axis
                output = solver(Ax, Bx(), x0=Ux, M=Mx) # INNER ITERATIONS
                if isinstance(output, tuple):  # If the solver returns the solution and some statistics
                    Ux = output[0]
                    inner_statistics_x.append(output[1])
                else:
                    Ux = output
                    inner_statistics_x.append(None)
                grad_Ux = s.grad(Ux)
                # Update the source terms
                By_t = s.source_transverse_y(grad_Ux, By_t)
                Bx_c = s.source_correction_x(grad_Ux, Bx_c)  
                
                # Solve the system of equations for y-axis
                output = solver(Ay, By(), x0=Uy, M=My) # INNER ITERATIONS
                if  isinstance(output, tuple): # If the solver returns the solution and some statistics
                    Uy = output[0]
                    inner_statistics_y.append(output[1])
                else: 
                    Uy = output
                    inner_statistics_y.append(None)
                grad_Uy = s.grad(Uy)
                # Update the source terms
                Bx_t = s.source_transverse_x(grad_Uy, Bx_t)
                By_c = s.source_correction_y(grad_Uy, By_c)
            
            ## END OF OUTER ITERATION ##
            
            outer_end_time = time()
            outer_times.append(outer_end_time-outer_start_time)
            
            # Calculate the residuals "on the fly" for early stopping
            on_fly_res['x'].append(residual(Ax, Ux, Bx()))
            on_fly_res['y'].append(residual(Ay, Uy, By()))
            on_fly_res_norm['y'].append(residual_norm(Ay, Uy, By()))
            on_fly_res_norm['x'].append(residual_norm(Ax, Ux, Bx()))
            
            # Store the displacement fields
            hist_Ux.append(Ux)
            hist_Uy.append(Uy)
            
            # Store the source vectors
            hist_Bx.append({'transverse' : Bx_t, 'boundary' : Bx_b, 'force' : Bx_f, 'correction' : Bx_c, 'all' : Bx()})
            hist_By.append({'transverse' : By_t, 'boundary' : By_b, 'force' : By_f, 'correction' : By_c, 'all' : By()})
                        
            # Trend between outer iterations
            trend['x'].append(np.linalg.norm(hist_Ux[-1]-hist_Ux[-2],1))
            trend['y'].append(np.linalg.norm(hist_Uy[-1]-hist_Uy[-2],1))
            
            # Print the progress of the iterations
            pbar.set_postfix_str(f'Normalized residuals: Rx {on_fly_res_norm["x"][-1]:.2e}, Ry {on_fly_res_norm["y"][-1]:.2e}, dRx {on_fly_res_norm["x"][-1]-on_fly_res_norm["x"][-2]:.2e}, dRy {on_fly_res_norm["y"][-1]-on_fly_res_norm["y"][-2]:.2e}')
    
            # Check early stopping criteria
            if step>1 and early_stopping:
                if not early_stopping_flag_x:
                    if trend['x'][-1] >= trend['x'][-2]: # Check if the trend is increasing
                        inc_trend_counter_x += 1 # Increment the counter
                    else:
                        inc_trend_counter_x = 0 # Reset the counter
                    if inc_trend_counter_x > inc_trend_counter_max: # Check if the counter is above the maximum
                        early_stopping_flag_x = True # Set the flag to stop the iterations
                if not early_stopping_flag_y:
                    if trend['y'][-1] >= trend['y'][-2]: # Check if the trend is increasing
                        inc_trend_counter_y += 1 # Increment the counter
                    else :
                        inc_trend_counter_y = 0 # Reset the counter
                    if inc_trend_counter_y > inc_trend_counter_max: # Check if the counter is above the maximum
                        early_stopping_flag_y = True # Set the flag to stop the iterations
                if early_stopping_flag_x and early_stopping_flag_y:
                    print('Early stopping after', step, 'iterations due to increasing trend')
                    break
            # Convergence critetion based on the trend
            if trend['x'][-1] < tol_trend and trend['y'][-1] < tol_trend: # Check if the trend is below the tolerance
                print('Solution converged after', step, 'iterations based on the trend of the displacement fields')
                break            
            # Convergence criterion on the fly (residuals)
            if on_fly_res['x'][-1] < tol_res and on_fly_res['y'][-1] < tol_res: # Check if the absolute residuals are below the tolerance
                print('Solution converged after', step, 'iterations based on the calculation of the residuals "on the fly"')
                break
            # Convergence criterion on the fly (normalized residuals)
            if on_fly_res_norm['x'][-1] < tol_res_norm and on_fly_res_norm['y'][-1] < tol_res_norm: 
                # Check if the normalized residuals are below the tolerance
                print('Solution converged after', step, 'iterations based on the calculation of the normalized residuals "on the fly"')
                break
            
        ### End of solver ###
        
        # Update the source terms that are not yet up-to-date
        By_c = s.source_correction_y(grad_Uy, By_c) 
        Bx_t = s.source_transverse_x(grad_Uy, Bx_t)

        # Store the residuals using the converged source vector
        for i in range(1,len(hist_Ux)):
            # Absolute residuals at outer iterations
            res['x'].append(residual(Ax, hist_Ux[i], Bx()))
            res['y'].append(residual(Ay, hist_Uy[i], By()))
            # Normalized residuals at outer iterations
            res_norm['x'].append(residual_norm(Ax, hist_Ux[i], Bx()))
            res_norm['y'].append(residual_norm(Ay, hist_Uy[i], By()))
        
        # Store the statistics
        statistics['trend'] = trend
        statistics['res'] = res
        statistics['res_norm'] = res_norm
        statistics['hist_Ux'] = np.array(hist_Ux)
        statistics['hist_Uy'] = np.array(hist_Uy)
        statistics['hist_Bx'] = hist_Bx
        statistics['hist_By'] = hist_By
        statistics['on_fly_res'] = on_fly_res
        statistics['on_fly_res_norm'] = on_fly_res_norm
        statistics['outer_iteration_times'] = outer_times
        statistics['inner_statistics'] = {'x' : inner_statistics_x, 'y' : inner_statistics_y}
        statistics['precond'] = precond.__name__
        statistics['precond_args'] = precond_args
            
        return Ux, Uy, statistics