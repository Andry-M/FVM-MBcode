# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 07/02/2025
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Preconditionners for the linear solvers implemented in MBcode
# -----------------------------------------------------------------------------

import scipy.sparse as sps

def spilu(A, **kwargs):
    """
        Incomplete LU factorization with fill-in.
        This is a wrapper for scipy.sparse.linalg.spilu.
        
        Parameters:
            - A (scipy.sparse matrix): The matrix to be factorized.
            - kwargs (dict): Additional keyword arguments for the factorization.
                - fill_factor (int): The fill factor for the factorization (ratio of non-zeros). Default is 10.
    """
    from scipy.sparse.linalg import spilu
    from scipy.sparse.linalg import LinearOperator
    fill_factor = kwargs.get('fill_factor', 10) # default value is 10 in scipy
    superLU = spilu(A, fill_factor=fill_factor)
    linear_operator = LinearOperator(A.shape, superLU.solve)
    return linear_operator

def ilu(A, **kwargs):
    """
        Incomplete LU factorization with fill-in.
        This is a wrapper for the inverse of ilupp.ILUTPreconditioner.
        
        Parameters:
            - A (scipy.sparse matrix): The matrix to be factorized.
            - kwargs (dict): Additional keyword arguments for the factorization.
                - fill_in (int): The fill in for the factorization (number of non-zeros per row). Default is 0.
                - threshold (float): Entries with a relative magnitude less than this are dropped. Default is 0.
    """
    from ilupp import ILUTPreconditioner
    fill_in = kwargs.get('fill_in', 100) # Zero fill-in is the default value in ilupp
    threshold = kwargs.get('threshold', 0.1) # Default value is 0 in ilupp
    prec = ILUTPreconditioner(A, fill_in=fill_in, threshold=threshold)
    L, U = prec.factors()
    def solve(x):
        y = sps.linalg.spsolve_triangular(L, x, lower=True)   # Solve L y1 = x
        y = sps.linalg.spsolve_triangular(U, y, lower=False)  # Solve U y = y1
        return y
    linear_operator = sps.linalg.LinearOperator(A.shape, matvec=solve)
    return linear_operator

def icholesky(A, **kwargs):
    """
        Incomplete Cholesky factorization with fill-in.
        This is a wrapper for the inverse of ilupp.ICholTPreconditioner.
        
        Parameters:
            - A (scipy.sparse matrix): The matrix to be factorized.
            - kwargs (dict): Additional keyword arguments for the factorization.
                - add_fill_in (int): The fill in for the factorization (number of non-zeros per column). Default is 0.
                - threshold (float): Entries with a relative magnitude less than this are dropped. Default is 0.
    """
    from ilupp import ICholTPreconditioner
    add_fill_in = kwargs.get('add_fill_in', 0) # Zero fill-in is the default value in ilupp
    threshold = kwargs.get('threshold', 0.) # Default value is 0 in ilupp
    prec = ICholTPreconditioner(A, add_fill_in=add_fill_in, threshold=threshold)
    L = prec.factors()[0]
    def solve(x):
        y = sps.linalg.spsolve_triangular(L, x, lower=True)   # Solve L y1 = x
        y = sps.linalg.spsolve_triangular(L.T, y, lower=False)  # Solve L^T y = y1
        return y
    linear_operator = sps.linalg.LinearOperator(A.shape, matvec=solve)
    return linear_operator