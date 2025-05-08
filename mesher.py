# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 28/11/2024
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Mesher for the finite volume method implemented in MBcode
# -----------------------------------------------------------------------------

# Libraries importation
import numpy as np                      # Array manipulation
import matplotlib.pyplot as plt         # Plots
from scipy.spatial import Delaunay      # Tesselation
from typing import List                 # Type specifications
from datetime import datetime           # Timestamp for saving plots
from os.path import isdir               # For folder test
from os import makedirs                 # For folder creation
from datetime import datetime           # Timestamp for saving plots
from mb_code.parameters import DTYPE    # Import the parameters

class Cell():
    """
        Store the cell information and its environment in the mesh.
    """
    def __init__(s, centroid, vertices):
        """
            Cell storage with the following attributes
            - centroid (int) : index of the centroid
            - vertices (np.array) : index of the vertices
            - stencil (dict) : inner stencil
            - bstencil (dict) : outer stencil
            - volume (float) : volume of the cell
            - grad_estimator (np.array) : matrix for explicit least square gradient calculation
            - grad_stencil (np.array) : stencil for explicit least square gradient calculation 
              
            Inner stencil (dict): Each key represents a neighboring cell
            - key : index of the neighboring cells
            - value : dict with the following keys\n
                - vertices (list) : index of the shared vertices
                - fcentroid (np.array) : coordinates of the face centroid
                - normal (np.array) : normal vector to the face
                - tangential (np.array) : tangential vector to the face
                - area (float) : area of the face
                - distance (np.array) : distance vector between the centroids
                - proj_distance (float) : projection of the distance vector on the normal
                - orth_correction (np.array) : correction vector for the orthogonality of the gradient (k)
                - skew_correction (np.array) : correction vector for the skewness of the gradient (m)
                - weight (float) : weight of the cell for gradient interpolation at the face centroids
            
            Outer stencil (dict): Each key represents a neighboring boundary point
            - key : index of the boundary point
            - value : dict with the following keys\n
                - vertices (list) : index of the boundary vertices
                - normal (np.array) : normal vector to the face
                - tangential (np.array) : tangential vector to the face
                - area (float) : area of the face
                - distance (np.array) : distance vector between the centroid and the boundary point
                - proj_distance (float) : projection of the distance vector on the normal
                - orth_correction (np.array) : correction vector for the orthogonality of the gradient (k)
                - bc_id (str) : identifier of the boundary condition
            
            Parameters:
                - centroid (int) : index of the centroid
                - vertices (np.array) : index of the vertices
        """
        # Known from the mesh creation
        s.centroid = centroid    # index of the centroid
        s.vertices = vertices    # index of the vertices
        
        # To be filled during mesh completion
        s.stencil = dict[int, dict[str, object]]()
        s.bstencil = dict[int, dict[str, object]]()
        s.grad_estimator : np.array = None
        s.grad_stencil : np.array = None
        s.volume : float = None
        
class Mesh2d():
    """
        2D mesh storage with the following attributes\n
        - centroids (np.array) : coordinates of the centroids
        - vertices (np.array) : coordinates of the vertices
        - bpoints (np.array) : coordinates of the boundary points
        - cells (list) : list of cells
        - boundaries (list) : list of boundary conditions
            
        The boundary conditions are defined as a list of dictionaries with the following keys\n
        - id : identifier of the boundary condition
        - condition : function that returns True if the condition is met, False otherwise
        TAKE CARE:  the condition must be defined in the form condition(v, w) where v and w are
                    the coordinates of the vertices. Strict equality is not recommended, use np.isclose() instead.
    """
    def __init__(s):
        s.centroids = []  
        s.vertices = [] 
        s.bpoints =  [] 
        s.cells : List[Cell]
        s.boundaries = []
        s.n_face_max = -1 # Maximum number of faces per cell
        
    def _generate_tri_mesh(s):   
        """
            Use the Delaunay tesselation algorithm to create a triangular mesh based on the vertices.
        """
        triangulation = Delaunay(s.vertices)
        centroids = []
        cells = []
        
        # Iterate over each element in the triangulation
        for i, simplex in enumerate(triangulation.simplices):
            # Get the coordinates of the vertices of the element
            simplex_points = s.vertices[simplex]
            # Calculate the centroid (average of the vertices)
            centroid = np.mean(simplex_points, axis=0)
            centroids.append(centroid)
            cells.append(Cell(i, simplex))
        
        s.cells = cells
        s.centroids = np.array(centroids, dtype=DTYPE)
        
    def _generate_stencils(s, n_face_max : int):
        """
            Generate the inner and outer (boundaries) stencils of the cells.\n
            Compute the least square estimator and stencil for the gradient.\n
            Calculate the volume of the cells.\n
            
            The value 7 of n_face_max allows to have a mix of triangles and quadrangles.
            
            Parameters:
                - n_face_max (int) : maximum number of faces per cell
        """
        s.n_face_max = n_face_max # Store the maximum number of faces per cell

        bpoints = [] # List of boundary points (dependant on the mesh)
        
        # Fill the inner and outer stencils
        for i, cell in enumerate(s.cells): # Loop over the cells
            n_face = len(cell.stencil.keys()) # Number of faces already registered for the cell
            
            # Inner stencil (between cells) 
            for j, other in enumerate(s.cells): # Loop over the other cells
                
                if i<j: # Avoid double counting
                    if n_face == n_face_max or (n_face_max==7 and n_face==4): # Stop the loop if the maximum number of faces is reached
                        break                      
                    
                    shared_vertices = np.intersect1d(cell.vertices, other.vertices)
                    if len(shared_vertices) == 2: # If the two cells share two vertices, they share a face
                        if not j in cell.stencil: # Avoid double counting
                            
                            # Calculate the relevant values to be stored in the stencil
                            xi, yi = s.centroids[cell.centroid] # Coordinates of the centroid of the cell i
                            xj, yj = s.centroids[other.centroid] # Coordinates of the centroid of the cell j
                            xv, yv = s.vertices[shared_vertices[0]] # Coordinates of the shared vertex v
                            xw, yw = s.vertices[shared_vertices[1]] # Coordinates of the shared vertex w
                            xm, ym = (xv+xw)/2, (yv+yw)/2 # Coordinates of the face centroid
                            normal = np.array([yv-yw, xw-xv]) # Normal vector to the face
                            if np.dot(normal, [xj-xi, yj-yi]) < 0: # Check the orientation of the normal
                                normal = -normal
                            normal /= np.linalg.norm(normal) # Normalize the normal vector
                            tangential = np.array([-normal[1], normal[0]]) # Tangential vector to the face
                            area = np.linalg.norm([xv-xw, yv-yw]) # Area of the face
                            distance = np.array([xj-xi, yj-yi]) # Distance vector between the centroids
                            proj_distance = np.dot(distance, normal) # Projection of the distance vector on the normal
                            delta = distance / proj_distance * area
                            orth_correction = area * normal - delta # Correction vector for the orthogonality of the gradient (k)
                            # Calculate the intercept between the shared vertices and the line between cells i and j
                            denominator = (xi - xj) * (yv - yw) - (yi - yj) * (xv - xw)
                            xf = ((xi * yj - yi * xj) * (xv - xw) - (xi - xj) * (xv * yw - yv * xw)) / denominator
                            yf = ((xi * yj - yi * xj) * (yv - yw) - (yi - yj) * (xv * yw - yv * xw)) / denominator
                            skew_correction = np.array([xm-xf, ym-yf]) # Correction vector for the skewness of the gradient (m)
                            weight = np.linalg.norm([xj-xf, yj-yf]) / np.linalg.norm(distance) # Weight for gradient interpolation at face centroid
                            
                            # Store the values in the stencil
                            dict_face_cell = {
                                'vertices' : shared_vertices,
                                'fcentroid' : np.array([xm, ym], dtype=DTYPE),
                                'normal' : normal,
                                'tangential' : tangential,
                                'area' : area,
                                'distance' : distance,
                                'proj_distance' : proj_distance,
                                'orth_correction' : orth_correction,
                                'skew_correction' : skew_correction,
                                'weight' : weight
                            }
                            dict_face_other = dict_face_cell.copy()
                            dict_face_other['normal'] = -normal
                            dict_face_other['tangential'] = -tangential
                            dict_face_other['distance'] = -distance
                            dict_face_other['orth_correction'] = -orth_correction
                            dict_face_other['weight'] = 1 - weight
                            
                            cell.stencil[other.centroid] = dict_face_cell
                            other.stencil[cell.centroid] = dict_face_other
                                           
                            n_face += 1 # Increment the number of faces already registered
            
            # Outer stencil (boundaries)
            for a, v in enumerate(cell.vertices): # Loop over the vertices of the cell
                for b, w in enumerate(cell.vertices): # Loop over the other vertices of the cell
                    if b>a: # Avoid double counting
                        for boundary in s.boundaries: # Loop over the boundaries of the mesh
                            if boundary['condition'](s.vertices[v], s.vertices[w]): # Check if the boundary condition is met
                                
                                # Calculate the relevant values to be stored in the stencil
                                xi, yi = s.centroids[cell.centroid] # Coordinates of the centroid of the cell i
                                xb, yb = (s.vertices[v]+s.vertices[w])/2 # Coordinates of the boundary point b
                                xv, yv = s.vertices[v] # Coordinates of the vertex v
                                xw, yw = s.vertices[w] # Coordinates of the vertex w
                                normal = np.array([yv-yw, xw-xv]) # Normal vector to the face
                                if np.dot(normal, [xb-xi, yb-yi]) < 0: # Check the orientation of the normal
                                    normal = -normal
                                normal /= np.linalg.norm(normal) # Normalize the normal vector
                                tangential = np.array([-normal[1], normal[0]]) # Tangential vector to the face
                                area = np.linalg.norm([xv-xw, yv-yw]) # Area of the face
                                distance = np.array([xb-xi, yb-yi]) # Distance vector between the centroid and the boundary point
                                proj_distance = np.dot(distance, normal) # Projection of the distance vector on the normal
                                delta = distance / proj_distance * area
                                orth_correction = area * normal - delta # Correction vector for the orthogonality of the gradient (k)
                                # By definition of the boundary point, the skewness correction is null
                                bc_id = boundary['id'] # Identifier of the boundary condition
                                
                                # Store the values in the stencil
                                bpoints.append(np.array([xb, yb])) # Add the boundary point to the mesh
                                dict_face_cell = {
                                    'vertices' : [v, w],
                                    'normal' : normal,
                                    'tangential' : tangential,
                                    'area' : area,
                                    'distance' : distance,
                                    'proj_distance' : proj_distance,
                                    'orth_correction' : orth_correction,
                                    'bc_id' : bc_id
                                }
                                cell.bstencil[len(bpoints)-1] = dict_face_cell
                                
                                break # Stop the loop over the boundary conditions   
            
            # Search for the gradient LSQ stencil
            grad_stencil = []
            for j, other in enumerate(s.cells): # Loop over the other cells
                #if i!=j: # Allowing self-comparison stabilizes the gradient estimation
                shared_vertices = np.intersect1d(cell.vertices, other.vertices)
                if len(shared_vertices) > 0: # If at least one vertice is shared
                    grad_stencil.append(j)
                if s.n_face_max == 3: # If elements are triangles
                    if len(grad_stencil)==25: # Maximum number of 16 cells (including the current one)
                        break # Stop to append grad_stencil
                elif s.n_face_max == 4: # If elements are quadrangles
                    if len(grad_stencil)==9: # Maximum number of 9 cells (including the current one)
                        break # Stop to append grad_stencil
                elif s.n_face_max == 7: # If elements are quadrangles and triangles
                    if len(grad_stencil)==16: # Maximum number of 16 cells (including the current one)
                        break # Stop to append grad_stencil
                    
            A = [] # Least square matrix
            for j in grad_stencil: # Loop over the neighboring cells
                other = s.cells[j]
                A.append(s.centroids[other.centroid] - s.centroids[cell.centroid])
                                    
            A = np.array(A, dtype=DTYPE) # Convert to numpy array
            grad_estimator = np.linalg.inv(A.T @ A) @ A.T # Calculate the estimator for explicit least square gradient calculation
            cell.grad_estimator = grad_estimator
            cell.grad_stencil = np.array(grad_stencil)
            
        # Compute the volume of the cells
        for i, cell in enumerate(s.cells): # Loop over the cells
            volume = 0
            vertices = s.vertices[cell.vertices]
            centroid = s.centroids[cell.centroid]
            # Sort the vertices to create a close counter-clockwise outline of the volume
            angles = []
            for v in vertices:
                angles.append(np.arctan2(v[1]-centroid[1], v[0]-centroid[0]))
            order = np.argsort(angles)
            vertices = vertices[order]
            # Add again the first vertices for the closing of the outline
            vertices = np.concatenate([vertices, vertices[0].reshape(1,2)], axis=0)
            # Compute the volume using the Shoelace formula
            for v in range(len(vertices)-1):
                volume += (vertices[v][0] * vertices[v+1][1] - vertices[v][1] * vertices[v+1][0])
            cell.volume = abs(volume)/2
            
        s.bpoints = np.array(bpoints, dtype=DTYPE) # Convert to numpy array
        
    def generate_bgrad_estimator(s, b_cond : dict):
        # Store the information for explicit least square gradient calculation
        for i, cell in enumerate(s.cells): # Loop over the cells
            # Search for the gradient LSQ stencil
            bgrad_stencil = []
            for j, other in enumerate(s.cells): # Loop over the other cells
                #if i!=j: # Allowing self-comparison stabilizes the gradient estimation
                shared_vertices = np.intersect1d(cell.vertices, other.vertices)
                if len(shared_vertices) > 0: # If at least one vertice is shared
                    bgrad_stencil.append(j)
                if s.n_face_max == 3: # If elements are triangles
                    if len(bgrad_stencil)==25: # Maximum number of 16 cells (including the current one)
                        break # Stop to append grad_stencil
                elif s.n_face_max == 4: # If elements are quadrangles
                    if len(bgrad_stencil)==9: # Maximum number of 9 cells (including the current one)
                        break # Stop to append grad_stencil
                elif s.n_face_max == 7: # If elements are quadrangles and triangles
                    if len(bgrad_stencil)==16: # Maximum number of 16 cells (including the current one)
                        break # Stop to append grad_stencil
                    
            A_x = [] # Least square matrix
            A_y = [] # Least square matrix
            for j in bgrad_stencil: # Loop over the neighboring cells
                other = s.cells[j]
                A_x.append(s.centroids[other.centroid] - s.centroids[cell.centroid])

            # Copy A and grad_stencil_x to A_y and grad_stencil_y
            A_y = A_x.copy()

            for b, face in cell.bstencil.items(): # Loop over the boundary points
                cdt_type_x = b_cond['x'][face['bc_id']]['type']
                cdt_type_y = b_cond['y'][face['bc_id']]['type']
                
                # x-axis boundary condition             
                if cdt_type_x == 'displacement' or cdt_type_x == 'Displacement':
                    A_x.append(s.bpoints[b] - s.centroids[cell.centroid])
                if cdt_type_y == 'displacement' or cdt_type_y == 'Displacement':
                    A_y.append(s.bpoints[b] - s.centroids[cell.centroid])
                                    
            A_x = np.array(A_x, dtype=DTYPE) # Convert to numpy array
            A_y = np.array(A_y, dtype=DTYPE) # Convert to numpy array
            bgrad_estimator_x = np.linalg.inv(A_x.T @ A_x) @ A_x.T # Calculate the estimator for explicit least square gradient calculation
            bgrad_estimator_y = np.linalg.inv(A_y.T @ A_y) @ A_y.T # Calculate the estimator for explicit least square gradient calculation
            cell.bgrad_estimator_x = bgrad_estimator_x
            cell.bgrad_estimator_y = bgrad_estimator_y
            cell.bgrad_stencil = np.array(bgrad_stencil)

    def plot(s, figsize=(8,8), filename : str = None):
        """
            Plot the primary mesh of the domain with centroids, vertices, faces and boundary points.\n
            
            Parameters:
                - figsize (tuple, default=(8,8)) : size of the figure
                - filename (str, default=None) : name of the file to save the plot. If None, no file is saved.
        """
        plt.figure(figsize=figsize)
        
        # Plot the vertices and centroids
        plt.scatter(s.vertices[:, 0], s.vertices[:, 1], color='b', s=10, label='Vertices')
        plt.scatter(s.centroids[:, 0], s.centroids[:, 1], color='r', s=20, label='Centroids')
        plt.scatter(s.bpoints[:, 0], s.bpoints[:, 1], color='g', s=10, label='Boundary points')
        
        # Draw the edges of the triangulation
        for cell in s.cells:
            for j, face in cell.stencil.items():
                v, w = face['vertices']
                x_coords = s.vertices[[v, w], 0]
                y_coords = s.vertices[[v, w], 1]
                plt.plot(x_coords, y_coords, color='gray', alpha=0.2)
            for b, face in cell.bstencil.items():
                v, w = face['vertices']
                x_coords = s.vertices[[v, w], 0]
                y_coords = s.vertices[[v, w], 1]
                plt.plot(x_coords, y_coords, color='black', alpha=0.2)

        # Setting the labels and title
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Mesh of the domain')
        plt.axis('equal')
        plt.legend(loc='best')
        
        # Save the plot
        if filename is not None:
            # Create the Results directory if it does not exist
            if not isdir('Results'):
                makedirs('Results')
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"Results/plot_mesh2d_{filename}_{timestamp}.png", dpi=500)

class BeamMesh2d(Mesh2d):
    """
        Define the mesh of a rectangular domain starting at (0,0).
        
        Boundaries of the domain are defined as follows:
        - west x=0
        - east x=xmax
        - south y=0
        - north y=ymax
    """
    def __init__(s, xmax : float, ymax : float, nx : int, ny : int):
        """
            Parameters:
                - xmax (float) : Domain size in x (width)
                - ymax (float) : Domain size in y (height)
                - nx (int) : Number of vertices in x direction
                - ny (int) : Number of vertices in y direction
        """
        super().__init__()
        s.xmax = xmax
        s.ymax = ymax
        s.nx = nx
        s.ny = ny
        s.boundaries = [
            {'id' : 'west', 'condition' : s.isWest},
            {'id' : 'east', 'condition' : s.isEast},
            {'id' : 'south', 'condition' : s.isSouth},
            {'id' : 'north', 'condition' : s.isNorth},
        ]
    
    def isWest(s, v, w):
            return np.isclose(v[0],0) and np.isclose(w[0],0)
    def isEast(s, v, w):
        return np.isclose(v[0],s.xmax) and np.isclose(w[0],s.xmax)
    def isSouth(s, v, w):
        return np.isclose(v[1],0) and np.isclose(w[1],0)
    def isNorth(s, v, w):
        return np.isclose(v[1],s.ymax) and np.isclose(w[1],s.ymax)
    
    def __generate_vertices(s):
        """
            Uniformly distribute the vertices of the primary mesh in the domain [[0,xmax],[0,ymax]].
        """
        vertices = []

        # Create points in Cartesian coordinates
        for yi in range(s.ny): # Line
            for xi in range(s.nx): # Column
                x = s.xmax/(s.nx-1) * xi
                y = s.ymax/(s.ny-1) * yi
                vertices.append((x, y))
        s.vertices = np.array(vertices, dtype=DTYPE) # Convert to numpy array
    
    def __generate_quad_mesh(s):
        """
            Create a regular quad mesh based on the regular vertices.
        """
        centroids = []
        cells = []

        # Loop through the grid to calculate centroids for each square cell
        cell_number = 0
        for yi in range(s.ny-1): # Line by line
            for xi in range(s.nx-1): # Column by column
                element_vertices_ind = [
                    yi * s.nx + xi,               # Bottom-left
                    yi * s.nx + (xi + 1),         # Bottom-right
                    (yi + 1) * s.nx + (xi + 1),   # Top-right
                    (yi + 1) * s.nx + xi          # Top-left
                ]
                element_vertices_coord = s.vertices[element_vertices_ind] # Coordinates of the vertices
                
                centroid = np.mean(element_vertices_coord, axis=0) # Calculate the centroid coordinates
                centroids.append(centroid)
                cells.append(Cell(cell_number, np.array(element_vertices_ind)))
                cell_number += 1
                
        s.cells = cells
        s.centroids = np.array(centroids, dtype=DTYPE)
    
    def init_quad(s):
        """
            Set up the mesh for quad elements.
        """
        s.__generate_vertices()
        s.__generate_quad_mesh()
        s._generate_stencils(n_face_max=4)
        
    def init_tri(s):
        """
            Set up the mesh for tri elements.
        """
        s.__generate_vertices()
        s._generate_tri_mesh()
        s._generate_stencils(n_face_max=3)
        
    def generate_grid(s, nx : int, ny : int):
        """
            Generate a regular grid of points for the interpolation.
            
            Parameters:
                - nx (int) : Number of points in x direction
                - ny (int) : Number of points in y direction
                
            Returns:
                - grid_x (np.array) : x-coordinates of the grid
                - grid_y (np.array) : y-coordinates of the grid
        """
        grid_x, grid_y = np.mgrid[0:s.xmax:nx*1j, 0:s.ymax:ny*1j] 
        return grid_x, grid_y

class BeamMeshHole2d(BeamMesh2d):
    """
        Define the mesh of a rectangular domain starting at (0,0) and containing a circular hole.
        
        Boundaries of the domain are defined as follows:
        - west x=0
        - east x=xmax
        - south y=0
        - north y=ymax
        - hole circular hole centered at (center) with radius (radius)
    """
    def __init__(s, xmax : float, ymax : float, nx : int, ny : int, center : np.array, 
                 radius : float, n_points_hole : int, n_layers_hole : int, 
                 layer_thickness : int, last_layer_distance : int):
        """
            Parameters:
                - xmax (float) : Domain size in x (width)
                - ymax (float) : Domain size in y (height)
                - nx (int) : Number of vertices in x direction
                - ny (int) : Number of vertices in y direction
                - center (np.array) : Coordinates of the center of the hole
                - radius (float) : Radius of the hole
                - n_points_hole (int) : Number of points on the hole boundary
                - n_layers_hole (int) : Number of layers on the hole boundary
                - layer_thickness (int) : Thickness of the layers
                - last_layer_distance (int) : Distance between the last layer and rest of the mesh
        """
        super().__init__(xmax, ymax, nx, ny)
        s.boundaries.append(
            {'id' : 'hole', 
             'condition' : s.isHole}
            )
        s.center = np.array(center, dtype=DTYPE)
        s.radius = radius
        s.n_points_hole = int(n_points_hole)
        s.n_layers_hole = int(n_layers_hole)
        s.layer_thickness = layer_thickness
        s.last_layer_distance = last_layer_distance
    
    def isHole(s, v, w):
        return np.isclose(np.linalg.norm(v-s.center), s.radius) and np.isclose(np.linalg.norm(w-s.center),s.radius)
        
    def __generate_vertices(s):
        """
            Generate the vertices of the primary mesh in the domain [[0,xmax],[0,ymax]] 
            with a circular hole of radius radius centered at s.center.
        """
        vertices = []
                
        # Distribute regularly points in the domain
        for yi in range(s.ny): # Line
            for xi in range(s.nx): # Column
                x = s.xmax/(s.nx-1) * xi
                y = s.ymax/(s.ny-1) * yi
                # Only add the point if it is not in the hole
                if (np.linalg.norm([x,y]-s.center) > s.radius + s.layer_thickness*s.n_layers_hole + s.last_layer_distance):
                    vertices.append(np.array([x, y]))
                    
        # Generate points on the hole boundary 
        theta = np.linspace(0, 2 * np.pi, s.n_points_hole, endpoint=False) # Angle discretization
        for layer in range(s.n_layers_hole+1): # Layer discretization
            for t in theta:
                vertex = s.center + (s.radius + s.layer_thickness*layer) * np.array([np.cos(t), np.sin(t)])
                # Only add the point if it is in the domain or if the outside points are accepted
                if (vertex[0] > 0 and vertex[0] < s.xmax and vertex[1] > 0 and vertex[1] < s.ymax):
                    vertices.append(vertex)
                        
        # Boundaries reconstruction
        # Add points at the intersection between the hole and other boundaries
        theta = np.linspace(0, 2 * np.pi, int(1e6), endpoint=False) # Brut-force method to find the intersection
        for layer in range(s.n_layers_hole):
            radius = s.radius + s.layer_thickness*layer
            # For x on the west and east boundaries
            for x in [0, s.xmax]:
                # If the x is on the hole boundary => y coordinate can be calculated
                if (radius**2 - (x-s.center[0])**2 >= 0):
                    y1 = s.center[1] + np.sqrt(radius**2 - (x-s.center[0])**2)
                    y2 = s.center[1] - np.sqrt(radius**2 - (x-s.center[0])**2)
                    for boundary in s.boundaries:
                        if boundary['id'] != 'hole': # For any boundary except the hole
                            # If the point (x, y1) is part of the boundary
                            if boundary['condition'](np.array([x, y1]), np.array([x, y1])) \
                                and y1 > 0 and y1 < s.ymax:
                                vertices.append(np.array([x, y1])) # Add the point
                            # If the point (x, y2) is part of the boundary
                            if boundary['condition'](np.array([x, y2]), np.array([x, y2])) \
                                and y2 > 0 and y2 < s.ymax:
                                vertices.append(np.array([x, y2])) # Add the point
            # For y on the south and north boundaries
            for y in [0, s.ymax]:
                # If the y is on the hole boundary => x coordinate can be calculated
                if (radius**2 - (y-s.center[1])**2 >= 0):
                    x1 = s.center[0] + np.sqrt(radius**2 - (y-s.center[1])**2)
                    x2 = s.center[0] - np.sqrt(radius**2 - (y-s.center[1])**2)
                    for boundary in s.boundaries:
                        if boundary['id'] != 'hole': # For any boundary except the hole
                            # If the point (x1, y) is part of the boundary
                            if boundary['condition'](np.array([x1, y]), np.array([x1, y])) \
                                and x1 > 0 and x1 < s.xmax:
                                vertices.append(np.array([x1, y])) # Add the point
                            # If the point (x2, y) is part of the boundary
                            if boundary['condition'](np.array([x2, y]), np.array([x2, y])) \
                                and x2 > 0 and x2 < s.xmax:
                                vertices.append(np.array([x2, y])) # Add the point
        
        s.vertices = np.array(vertices, dtype=DTYPE) # Convert to numpy array
        
    def _generate_tri_mesh(s):   
        """
            Use the Delaunay tesselation algorithm to create a triangular mesh based on the vertices\n
            Remove the cells and the centroids generated inside the hole.\n
            There should be no vertex inside the hole as the __generate_vertices() method takes care of it.
        """
        super()._generate_tri_mesh()
        
        # Remove the cells inside the hole
        i = 0
        while i < len(s.cells): # Loop over the cells
            # Check if the cell centroid is inside the hole
            if np.linalg.norm(s.centroids[s.cells[i].centroid] - s.center) < s.radius:
                s.centroids = np.delete(s.centroids, s.cells[i].centroid, axis=0) # Remove the centroid
                s.cells.pop(i) # Remove the cell
                # Shift the indices of the cells and centroids after the removed ones
                for cell in s.cells:
                    for j in cell.stencil.keys():
                        if j > i:
                            cell.stencil[str(int(j)-1)] = cell.stencil.pop(j)
                            #cell.stencil['cell'][j] -= 1
                    if cell.centroid > i:
                        cell.centroid -= 1
            else:
                i += 1
        
    def init_tri(s):
        """
            Set up the mesh for tri elements.
        """
        s.__generate_vertices()
        s._generate_tri_mesh()
        s._generate_stencils(n_face_max=3)
        
    def init_quad(s):
        """
            Set up the mesh for quad elements.\n
            First a tri mesh is created and post-processed to merge most of the triangles into quads.\n
            The meshes created with this function must be visualized to check the quality of the quads.\n
            
            The merging is done by comparing the aspect ratio of the candidate quadrangle with the mean aspect
            ratio of the candidate merged triangles.
        """
        s.__generate_vertices()
        s._generate_tri_mesh()
        s._generate_stencils(n_face_max=3)

        # Merge the triangles into quads
        new_cells = [] # New list of cells after merging
        new_centroids = [] # New list of centroids after merging
        merged = set() # Set of already merged cells to avoid multiple merges

        for i, cell in enumerate(s.cells): # Loop over the cells
            if i in merged: # Skip the cell if it has already been merged
                continue
            merged_cells = [cell] # By default the cell is merged with itself
            merged_centroid = s.centroids[cell.centroid] # By default the centroid is the same as the cell centroid
            
            # Calculate the aspect ratio of the cell
            old_vertices_coords = s.vertices[cell.vertices]
            old_lengths = [np.linalg.norm(old_vertices_coords[i] - old_vertices_coords[j]) 
                            for i in range(3) for j in range(i+1, 3)] # Distance between each pair of vertices
            old_lengths = np.sort(old_lengths)[:-1] # Remove the hypotenus
            old_aspect_ratio = np.max(old_lengths) / np.min(old_lengths)
            candidates = []
            
            # Store the boudaries of the cell to avoid merging two cells with a face on the same boundary
            boundaries_i = [face['bc_id'] for face in cell.bstencil.values()]
            
            for j in cell.stencil.keys(): # Loop over the cells in the stencil
                if j in merged: # Skip the stencil cell if it has already been merged
                    continue
                
                # Check that the two cells do not have a face on the same boundary
                boundaries_j = [face['bc_id'] for face in s.cells[j].bstencil.values()]
                if len(np.intersect1d(boundaries_i, boundaries_j)) > 0: 
                    continue # Skip the stencil cell if it shares a boundary with the current cell
                
                other_cell = s.cells[j] # Get the other Cell

                # Calculate the aspect ratio of the candidate merged cell
                candidate_cell_vertices = np.unique(np.concatenate([c.vertices for c in [cell, other_cell]]))
                candidate_cell_vertices_coords = s.vertices[candidate_cell_vertices]
                candidate_lengths = [np.linalg.norm(candidate_cell_vertices_coords[i] - candidate_cell_vertices_coords[j]) 
                            for i in range(4) for j in range(i+1, 4)] # Distance between each pair of vertices
                candidate_lengths = np.sort(candidate_lengths)[:-2] # remove the diagonals
                aspect_ratio = np.max(candidate_lengths) / np.min(candidate_lengths)
                
                # Calculate the aspect ratio of the other cell
                old_other_vertices_coords = s.vertices[other_cell.vertices]
                old_other_lengths = [np.linalg.norm(old_other_vertices_coords[i] - old_other_vertices_coords[j]) 
                            for i in range(3) for j in range(i+1, 3)] # Distance between each pair of vertices
                old_other_lengths = np.sort(old_other_lengths)[:-1] # Remove the hypotenus
                old_other_aspect_ratio = np.max(old_other_lengths) / np.min(old_other_lengths)
                
                # If the aspect ratio of the candidate merged cell is better 
                # than the mean aspect ratio of the two cells, save the candidate
                target = np.mean([old_aspect_ratio,old_other_aspect_ratio])
                if (aspect_ratio <= target):
                    candidates.append([j, aspect_ratio])
            
            candidates = np.array(candidates)
            
            if len(candidates) > 0: # If there are candidates
                # Add the best candidate to the merged cells
                index_best_aspect_ratio = np.argmin(candidates[:,1])
                j = int(candidates[index_best_aspect_ratio][0])
                other_cell = s.cells[j] # Get the other Cell
                candidate_cell_vertices = np.unique(np.concatenate([c.vertices for c in [cell, other_cell]]))
                candidate_cell_vertices_coords = s.vertices[candidate_cell_vertices]
                merged_cells.append(other_cell) # Add the other cell to the merged cells
                merged_centroid = np.mean(candidate_cell_vertices_coords, axis=0) # Update the centroid
                merged.add(j) # Add the other cell to the merged set
            
            # Append the new cell and centroid lists
            new_cell_vertices = np.unique(np.concatenate([c.vertices for c in merged_cells]))
            new_cells.append(Cell(len(new_centroids), new_cell_vertices))
            new_centroids.append(merged_centroid)
            merged.add(i) # Add the current cell to the merged set

        s.cells = new_cells # Update the cells
        s.centroids = np.array(new_centroids, dtype=DTYPE) # Update the centroids
        s._generate_stencils(n_face_max=7) # Generate the stencils for triangles + quadrangles
        
    def generate_grid(s, nx : int, ny : int):
        """
            Generate a regular grid of points for the interpolation.\n
            
            Parameters:
                - nx (int) : Number of points in x direction
                - ny (int) : Number of points in y direction
                
            Returns:
                - grid_x (np.array) : x-coordinates of the grid
                - grid_y (np.array) : y-coordinates of the grid
        """
        grid_x, grid_y = super().generate_grid(nx, ny)
        # Remove the points inside the hole
        grid_points = np.array([grid_x.flatten(), grid_y.flatten()]).T # Convert to (point, x, y) format
        grid_points = grid_points[np.linalg.norm(grid_points-s.center, axis=1) > s.radius]
        # Add points on the hole boundary
        n_points_hole = int(nx * 4 * s.radius / s.xmax + ny * 4 * s.radius / s.ymax) # Number of points on the hole boundary
        # The number of points on the hole boundary is proportional to the number of points in the grid and was tuned arbitrarily
        theta = np.linspace(0, 2 * np.pi, n_points_hole, endpoint=False) # Angle discretization
        for t in theta: # Loop over the angles
            grid_points = np.append(grid_points, [s.center + s.radius * np.array([np.cos(t), np.sin(t)])], axis=0)
        return grid_points[:,0], grid_points[:,1]
       
