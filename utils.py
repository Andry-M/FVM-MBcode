# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 28/11/2024
# Context: INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Post-processing functions for the Finite Volume Method implemented in MBcode
# -----------------------------------------------------------------------------

from mb_code.mesher import Mesh2d, BeamMeshHole2d
from multimethod import multimethod

# Libraries importation
import numpy as np                                              # Array manipulation
import matplotlib.pyplot as plt                                 # Plots
from matplotlib.animation import FuncAnimation, FFMpegWriter    # Animation
from scipy.interpolate import RBFInterpolator                   # Interpolation for plots
from typing import List, Callable                               # Type specifications
from datetime import datetime                                   # Timestamp for saving plots
from os.path import isdir                                       # For folder test
from os import makedirs                                         # For folder creation
from datetime import datetime                                   # Timestamp for saving plots

# Set the path to the ffmpeg executable for the mp4 files generation
plt.rcParams['animation.ffmpeg_path'] = 'D:/ffmpeg/bin/ffmpeg.exe'    

def residual(A, U, B) -> float:
    """
        Calculate the residual of the linear system AU = B.\n
        The residual is defined as the L1 norm of the difference between AU and B.
        
        Parameters:
            - A (np.array) : Matrix A
            - U (np.array) : Solution vector U
            - B (np.array) : Right-hand side vector B
    """
    return np.linalg.norm(A @ U - B, 1)

def residual_norm(A, U, B) -> float:
    """
        Calculate the normalized residual of the linear system AU = B as defined in OpenFOAM.\n
        The normalized residual is defined as the L1 norm of the difference between AU and B
        divided by the sum of the L1 norms of AU - A x mean(U) and B - A x mean(U).
        
        Parameters:
            - A (np.array) : Matrix A
            - U (np.array) : Solution vector U
            - B (np.array) : Right-hand side vector B
    """
    num = np.linalg.norm(A @ U - B, 1)
    Um = np.ones_like(U) * U.mean()
    den = np.linalg.norm(A @ U - A @ Um,1) + np.linalg.norm(B - A @ Um,1)
    if den==0:
        return None # Avoid division by zero
    else:
        return num/den

def residual_map(A, U, B) -> float:
    """
        Calculate the residual map of the linear system AU = B.\n
        The residual is defined as the absolute difference between AU and B.
        
        Parameters:
            - A (np.array) : Matrix A
            - U (np.array) : Solution vector U
            - B (np.array) : Right-hand side vector B
    """
    return A @ U - B

def residual_norm_map(A, U, B) -> float:
    """
        Calculate the normalized residual of the linear system AU = B as defined in OpenFOAM.\n
        The normalized residual map is defined as the absolute difference between AU and B
        divided by the sum of |AU - A x mean(U)| + |B - A x mean(U)|.
        
        Parameters:
            - A (np.array) : Matrix A
            - U (np.array) : Solution vector U
            - B (np.array) : Right-hand side vector B
    """
    num = A @ U - B
    Um = np.ones_like(U) * U.mean()
    den = np.linalg.norm(A @ U - A @ Um,1) + np.linalg.norm(B - A @ Um,1)
    if den==0:
        return num # Avoid division by zero
    else:
        return num/den
    
def diff_map(U, U_ref) -> np.array:
    """
        Calculate the difference map of the displacement field U with respect to a reference field U_ref.\n
        The map is defined as the absolute difference between U and U_ref.
        
        Parameters:
            - U (np.array) : Displacement field U
            - U_ref (np.array) : Reference displacement field U_ref
    """
    return np.asarray(U) - np.asarray(U_ref)

@multimethod  
def plot_history_fvm(trend : dict, res : dict, res_norm : dict, filename = None):
    """
        Plot the history of the convergence criteria for the Finite Volume Method.
        
        Parameters:
            - trend (dict) : difference between consecutive outer iterations
            - res (dict) : absolute residuals at outer iterations
            - res_norm (dict) : normalized residuals at outer iterations
            - filename (str, default=None) : name of the file to save the plot. If None, no file is saved.
    """
    filename = str(filename)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    # Trend
    axs[0].plot(trend['x'], label='x-axis', color='darkblue', linewidth=1.5)
    axs[0].plot(trend['y'], label='y-axis', color='coral', linewidth=1.5)
    axs[0].set_title('Difference between consecutive outer iterations', fontsize=16)
    axs[0].set_xlabel('Iteration', fontsize=15)
    axs[0].set_ylabel(r'$U_{n}-U_{n-1}$', fontsize=15)
    axs[0].legend(fontsize=16)
    axs[0].set_yscale('log')
    axs[0].grid()
    # Residual
    axs[1].plot(res['x'], label='x-axis', color='darkblue', linewidth=1.5)
    axs[1].plot(res['y'], label='y-axis', color='coral', linewidth=1.5)
    axs[1].set_title('Absolute residual', fontsize=16)
    axs[1].set_xlabel('Iteration', fontsize=15)
    axs[1].set_ylabel(r'$||AU-B||_1$', fontsize=15)
    axs[1].legend(fontsize=16)
    axs[1].set_yscale('log')
    axs[1].grid()
    # Normalized residual
    axs[2].plot(res_norm['x'], label='x-axis', color='darkblue', linewidth=1.5)
    axs[2].plot(res_norm['y'], label='y-axis', color='coral', linewidth=1.5)
    axs[2].set_title('Normalized residual (OpenFOAM)', fontsize=16)
    axs[2].set_xlabel('Iteration', fontsize=15)
    axs[2].set_ylabel(r'$\frac{||AU-B||_1}{||AU-A\bar{U}||_1+||B-A\bar{U}||_1}$', fontsize=15)
    axs[2].legend(fontsize=16)
    axs[2].set_yscale('log')
    axs[2].grid()
    # Save the plot
    if filename is not None:
        if not isdir('Results'):
            makedirs('Results')
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Results/plot_history_{filename}_{timestamp}.png", dpi=500)

@multimethod
def plot_history_fvm(trend_x : List, 
                     trend_y : List,
                     res_x : List,
                     res_y : List,
                     res_norm_x : List,
                     res_norm_y : List,
                     filename : str):
    plot_history_fvm({'x': trend_x, 'y': trend_y},
                     {'x': res_x, 'y': res_y},
                     {'x': res_norm_x, 'y': res_norm_y},
                     filename)

@multimethod
def plot_history_fvm(trend_x : List, 
                     trend_y : List,
                     res_x : List,
                     res_y : List,
                     res_norm_x : List,
                     res_norm_y : List):
    plot_history_fvm({'x': trend_x, 'y': trend_y},
                     {'x': res_x, 'y': res_y},
                     {'x': res_norm_x, 'y': res_norm_y},
                     None)
        
def plot_point_solution_imshow(points : np.array, Ux : np.array, Uy : np.array, field : str = None, filename : str = None):
    """
        Plot the displacement field on the corresponding grid of points using matplotlib.pyplot.imshow.\n
        The displacement field is represented by the color of the points.\n
        
        Parameters:
            - points (np.array) : grid of points
            - Ux (np.array) : x-axis displacement field
            - Uy (np.array) : y-axis displacement field
            - field (str, default=None) : name of the field to plot
            - filename (str, default=None) : name of the file to save the plot. If None, no file is saved.
    """
    points = np.array(points)
    
    # Determine the bounds of the grid
    x_min, x_max = points[..., 0].min(), points[..., 0].max()
    y_min, y_max = points[..., 1].min(), points[..., 1].max()

    # Calculate grid resolution
    x_unique = np.unique(points[..., 0])
    y_unique = np.unique(points[..., 1])
    x_grid_size = len(x_unique)
    y_grid_size = len(y_unique)

    # Create a 2D grid and fill it with NaN
    Ux_grid = np.full((y_grid_size, x_grid_size), np.nan)
    Uy_grid = np.full((y_grid_size, x_grid_size), np.nan)

    # Fill the grid with the provided values
    for (x, y), ux, uy in zip(points, Ux, Uy):
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]
        Ux_grid[y_idx, x_idx] = ux
        Uy_grid[y_idx, x_idx] = uy

    # Plot the grid using imshow
    fig, axs = plt.subplots(1, 2, figsize=(18, 5), tight_layout=True)
    imshow_x = axs[0].imshow(Ux_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', aspect='equal')
    axs[0].set_title(f'x-axis {field}', fontsize=16)
    axs[0].set_xlabel('X-axis', fontsize=15)
    axs[0].set_ylabel('Y-axis', fontsize=15)
    axs[0].axis('equal')
    axs[0].set_xlim(0, np.ceil(x_max));
    axs[0].set_ylim(0, np.ceil(y_max));
    cbar_x = plt.colorbar(imshow_x, ax=axs[0], orientation='vertical', pad=0.01)
    cbar_x.ax.tick_params(labelsize=13)  # Increase font size of colorbar values

    
    imshow_y = axs[1].imshow(Uy_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', aspect='equal')
    axs[1].set_title(f'y-axis {field}', fontsize=16)
    axs[1].set_xlabel('X-axis', fontsize=15)
    axs[1].set_ylabel('Y-axis', fontsize=15)
    axs[1].axis('equal')
    axs[1].set_xlim(0, np.ceil(x_max));
    axs[1].set_ylim(0, np.ceil(y_max));
    cbar_y = plt.colorbar(imshow_y, ax=axs[1], orientation='vertical', pad=0.01)
    cbar_y.ax.tick_params(labelsize=13)  # Increase font size of colorbar values
    
    # Save the plot
    if filename is not None:
        if not isdir('Results'):
            makedirs('Results')
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Results/plot_solution_{filename}_{timestamp}.png", dpi=500)

def plot_point_solution_scatter(points : np.array, Ux : np.array, Uy : np.array, field : str = None, filename : str = None):
    """
        Plot the displacement field on the corresponding grid of points using matplotlib.pyplot.scatter.\n
        The displacement field is represented by the color of the points.\n
        
        Parameters:
            - points (np.array) : grid of points
            - Ux (np.array) : x-axis displacement field
            - Uy (np.array) : y-axis displacement field
            - field (str, default=None) : name of the field to plot
            - filename (str, default=None) : name of the file to save the plot. If None, no file is saved.
    """
    # Create a grid for interpolation
    fig, axs = plt.subplots(1, 2, figsize=(18, 5), tight_layout=True)
    # x-axis
    scatter_x = axs[0].scatter(points[...,0], points[...,1], c=Ux, cmap='viridis')
    axs[0].set_title(f'x-axis {field}', fontsize=16)
    axs[0].set_xlabel('X-axis', fontsize=15)
    axs[0].set_ylabel('Y-axis', fontsize=15)
    axs[0].axis('equal')
    cbar_x = plt.colorbar(scatter_x, ax=axs[0], orientation='vertical', pad=0.01)
    cbar_x.ax.tick_params(labelsize=13)  # Increase font size of colorbar values
    # y-axis
    scatter_y = axs[1].scatter(points[...,0], points[...,1], c=Uy, cmap='viridis')
    axs[1].set_title(f'y-axis {field}', fontsize=16)
    axs[1].set_xlabel('X-axis', fontsize=15)
    axs[1].set_ylabel('Y-axis', fontsize=15)
    axs[1].axis('equal')
    cbar_y = plt.colorbar(scatter_y, ax=axs[1], orientation='vertical', pad=0.01)
    cbar_y.ax.tick_params(labelsize=13)  # Increase font size of colorbar values
    # Save the plot
    if filename is not None:
        if not isdir('Results'):
            makedirs('Results')
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Results/plot_solution_{filename}_{timestamp}.png", dpi=500)

def get_evaluator(points : np.array, field : np.array):
    """
        Create an evaluator for the field using linear interpolation.\n
        The evaluator is created using <b>scipy.interpolate.RBFInterpolator</b>.
        
        Parameters:
            - points (np.array) : grid of points
            - field (np.array) : field to interpolate
        
        Returns:
            - field_interp (RBFInterpolator) : evaluator for the field
    """
    # Create Radial Basis Function interpolators
    field_interp = RBFInterpolator(points, field, kernel='linear')
    
    # Interpolate the field on the grid
    return field_interp

def interpolate_solution(mesh : Mesh2d, Ux : np.array, Uy : np.array, nx : int, ny : int):
    """
        Interpolate the displacement field on a grid of points using linear interpolation.\n
        Use get_evaluator() to create the interpolator for the field.
        
        Parameters:
            - mesh (Mesh2d) : mesh of the domain
            - Ux (np.array) : x-axis displacement field
            - Uy (np.array) : y-axis displacement field
            - nx (int) : number of points in x direction
            - ny (int) : number of points in y direction
            
        Returns:
            - grid (np.array, shape = (#points,2)) : grid of points 
            - grid_Ux (np.array) : interpolated x-axis displacement field
            - grid_Uy (np.array) : interpolated y-axis displacement field
    """
    centroids = [mesh.centroids[c.centroid] for c in mesh.cells]

    # Create Radial Basis Function interpolators
    Ux_interp = get_evaluator(centroids, Ux)
    Uy_interp = get_evaluator(centroids, Uy)
    
    # Generate the grid of points
    grid_x, grid_y = mesh.generate_grid(nx, ny)
    grid = np.moveaxis([grid_x.flatten(), grid_y.flatten()], 0, -1)
    
    # Interpolate the displacement field on the grid
    grid_Ux = Ux_interp(grid)
    grid_Uy = Uy_interp(grid)
    
    return grid, grid_Ux, grid_Uy

def plot_interpolated_solution(mesh : Mesh2d, Ux : np.array, Uy : np.array, nx : int, 
                               ny : int, field : str = None, filename : str = None):
    """
        Plot the displacement field on a grid of points using linear interpolation.\n
        Use interpolate_solution() to interpolate the field on the grid.
        
        Parameters:
            - mesh (Mesh2d) : mesh of the domain
            - Ux (np.array) : x-axis displacement field
            - Uy (np.array) : y-axis displacement field
            - nx (int) : number of points in x direction
            - ny (int) : number of points in y direction
            - field (str, default=None) : name of the field to plot
            - filename (str, default=None) : name of the file to save the plot. If None, no file is saved.
    """    
    grid, grid_Ux, grid_Uy = interpolate_solution(mesh, Ux, Uy, nx, ny)

    # Plot the interpolated solution
    if type(mesh).__name__ == BeamMeshHole2d.__name__:
        plot_point_solution_scatter(grid, grid_Ux, grid_Uy, field, filename)
    else:
        plot_point_solution_imshow(grid, grid_Ux, grid_Uy, field, filename)

def animate_convergence_2d(mesh : Mesh2d, res : dict, res_norm : dict, hist_Ux : List, hist_Uy : List,
                           nx : int, ny : int, fps : int, xlabel_delta : int, frames, save_path : str):
    """
        Create an animation of the convergence of the Finite Volume Method.\n
        The animation shows the evolution of the absolute residual, normalized residual, and the interpolated displacement field.\n
        The animation is saved in a mp4 file using the FFMpeg writer.
        
        Parameters:
            - mesh (Mesh2d) : mesh of the domain
            - res (dict) : absolute residuals at each iteration
            - res_norm (dict) : normalized residuals at each iteration
            - hist_Ux (List) : history of the x-axis displacement field
            - hist_Uy (List) : history of the y-axis displacement field
            - nx (int) : number of points in x direction
            - ny (int) : number of points in y direction
            - fps (int) : frames per second of the animation
            - xlabel_delta (int) : interval between x-axis labels
            - frames (iterable) : iterable of frames in the animation
            - save_path (str) : path to save the mp4 file
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), tight_layout=True)

    # Set up initial plots for the residuals    
    # Residual
    axs[0, 0].set_title('Absolute residual', fontsize=16)
    axs[0, 0].set_xlabel('Iteration', fontsize=15)
    axs[0, 0].set_ylabel(r'$||AU-B||_1$', fontsize=15)
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid()
    axs[0, 0].set_xlim(0, len(res['x']))
    axs[0, 0].set_xticks(ticks=[0]+list(range(xlabel_delta-1, len(res['x'])+1,xlabel_delta)), 
                            labels=[1] + list(range(xlabel_delta, len(res['x'])+1,xlabel_delta)))
    axs[0, 0].set_ylim(min(res['x'] + res['y'])/2, max(res['x'] + res['y'])*2)
    res_x, = axs[0, 0].plot([], [], label='x-axis', color='darkblue', linewidth=1.5)
    res_y, = axs[0, 0].plot([], [], label='y-axis', color='coral', linewidth=1.5)
    axs[0, 0].legend(fontsize=16)
    # Normalized residual
    axs[0, 1].set_title('Normalized residual (OpenFOAM)', fontsize=16)
    axs[0, 1].set_xlabel('Iteration', fontsize=15)
    axs[0, 1].set_ylabel(r'$\frac{||AU-B||_1}{||AU-A\bar{U}||_1+||B-A\bar{U}||_1}$', fontsize=15)
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid()
    axs[0, 1].set_xlim(0, len(res_norm['x']))
    axs[0, 1].set_xticks(ticks=[0]+list(range(xlabel_delta-1, len(res_norm['x'])+1,xlabel_delta)), 
                            labels=[1] + list(range(xlabel_delta, len(res_norm['x'])+1,xlabel_delta)))
    axs[0, 1].set_ylim(min(res_norm['x'] + res_norm['y'])/2, max(res_norm['x'] + res_norm['y'])*2)
    res_norm_x, = axs[0, 1].plot([], [], label='x-axis', color='darkblue', linewidth=1.5)
    res_norm_y, = axs[0, 1].plot([], [], label='y-axis', color='coral', linewidth=1.5)
    axs[0, 1].legend(fontsize=16)

    # Initial plot for interpolated displacement
    # Prepare grids for interpolation
    grid_x, grid_y = mesh.generate_grid(nx, ny)
    points = np.moveaxis([grid_x.flatten(), grid_y.flatten()], 0, -1)
    grid_Ux = None
    grid_Uy = None
    
    if type(mesh).__name__ != BeamMeshHole2d.__name__:
        # Calculate grid resolution for imshow
        x_unique = np.unique(points[..., 0])
        y_unique = np.unique(points[..., 1])
        x_grid_size = len(x_unique)
        y_grid_size = len(y_unique)
        
        # Create a 2D grid and fill it with NaN
        grid_Ux = np.full((y_grid_size, x_grid_size), np.nan)
        grid_Uy = np.full((y_grid_size, x_grid_size), np.nan)
    
    # x-axis
    if type(mesh).__name__ == BeamMeshHole2d.__name__:
        displ_x = axs[1,0].scatter(points[...,0], points[...,1], c=np.zeros((nx, ny)),
                                cmap='viridis', vmin=np.min(hist_Ux), vmax=np.max(hist_Ux))
    else :
        displ_x = axs[1,0].imshow(grid_Ux, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                            origin='lower', cmap='viridis', aspect='equal', vmin=np.min(hist_Ux), vmax=np.max(hist_Ux))
    axs[1,0].set_title('x-axis displacement field', fontsize=16)
    axs[1,0].set_xlabel('X-axis', fontsize=15)
    axs[1,0].set_ylabel('Y-axis', fontsize=15)
    axs[1,0].axis('equal')
    axs[1,0].grid()
    
    cbar_x = plt.colorbar(displ_x, ax=axs[1,0], orientation='vertical', pad=0.01)
    cbar_x.ax.tick_params(labelsize=13)  # Increase font size of colorbar values
    ticks = np.linspace(np.min(hist_Ux), np.max(hist_Ux), num=7)
    cbar_x.set_ticks(ticks)
    
    # y-axis
    if type(mesh).__name__ == BeamMeshHole2d.__name__:
        displ_y = axs[1,1].scatter(points[...,0], points[...,1], c=np.zeros((nx, ny)), 
                                cmap='viridis', vmin=np.min(hist_Uy), vmax=np.max(hist_Uy))
    else :
        displ_y = axs[1,1].imshow(grid_Uy, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                            origin='lower', cmap='viridis', aspect='equal', vmin=np.min(hist_Uy), vmax=np.max(hist_Uy))
    axs[1,1].set_title('y-axis displacement field', fontsize=16)
    axs[1,1].set_xlabel('X-axis', fontsize=15)
    axs[1,1].set_ylabel('Y-axis', fontsize=15)
    axs[1,1].axis('equal')
    axs[1,1].grid()

    cbar_y = plt.colorbar(displ_y, ax=axs[1,1], orientation='vertical', pad=0.01)
    cbar_y.ax.tick_params(labelsize=13)  # Increase font size of colorbar values
    ticks = np.linspace(np.min(hist_Uy), np.max(hist_Uy), num=7)
    cbar_y.set_ticks(ticks)
    
    def init():
        return (res_x, res_y, res_norm_x, res_norm_y, displ_x, displ_y)

    # Update function for animation
    def update(i):  
        # Update absolute residual plot
        res_x.set_data(range(i), res['x'][:i])
        res_y.set_data(range(i), res['y'][:i])

        # Update normalized residual plot
        res_norm_x.set_data(range(i), res_norm['x'][:i])
        res_norm_y.set_data(range(i), res_norm['y'][:i])
        
        # Create Radial Basis Function interpolators
        Ux_interp = RBFInterpolator(mesh.centroids, hist_Ux[i], kernel='linear')
        Uy_interp = RBFInterpolator(mesh.centroids, hist_Uy[i], kernel='linear')
        
        # Interpolate the displacement field on the grid
        Ux = Ux_interp(points)
        Uy = Uy_interp(points)

        # Update displacement plots with dynamic color limits
        if type(mesh).__name__ == BeamMeshHole2d.__name__:
            displ_x.set_array(Ux.flatten())
            displ_y.set_array(Uy.flatten())
        else:            
            # Fill the grid with the provided values
            for p, ux, uy in zip(points, Ux, Uy):
                x, y = p[0], p[1]
                x_idx = np.where(x_unique == x)[0][0]
                y_idx = np.where(y_unique == y)[0][0]
                grid_Ux[y_idx, x_idx] = ux
                grid_Uy[y_idx, x_idx] = uy
            displ_x.set_data(grid_Ux)
            displ_y.set_data(grid_Uy)
        
        return (res_x, res_y, res_norm_x, res_norm_y, displ_x, displ_y)

    # Create the animation
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)

    # Save to mp4 using ffmpeg writer 
    animation.save(save_path, writer=FFMpegWriter(fps=fps))
    
    # Close the figure to not show it in the notebook
    plt.close();