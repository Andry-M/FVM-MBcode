# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 07/02/2025
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Linear solvers for the finite volume method implemented in MBcode
# -----------------------------------------------------------------------------

from numpy import isnan
from numpy.linalg import norm
from mb_code.parameters import RTOL, ATOL, MAXITER

def __solver_core(solver, A, b, x0, M, rtol, atol, maxiter):
    iters = 0
    def nonlocal_iterate(xk):
        if isnan(norm(xk)):
            raise ValueError('Linear solver ended due to uncontrolled solution norm. Try to modify the preconditionner or the solver.')
        nonlocal iters
        iters+=1
    U, info = solver(A, b, x0=x0, M=M, callback=nonlocal_iterate, rtol=rtol, atol=atol, maxiter=maxiter)
    return U, info, iters

def scipy_sparse_bicgstab(A, b, **kwargs):
    from scipy.sparse.linalg import bicgstab
    x0 = kwargs.get('x0', None)
    M = kwargs.get('M', None)
    rtol = kwargs.get('rtol', RTOL)
    atol = kwargs.get('atol', ATOL)
    maxiter = kwargs.get('maxiter', MAXITER)
    U, info, iters = __solver_core(bicgstab, A, b, x0, M, rtol, atol, maxiter)
    return U, {'info' : info, 'iterations': iters}

def scipy_sparse_cg(A, b, **kwargs):
    from scipy.sparse.linalg import cg
    x0 = kwargs.get('x0', None)
    M = kwargs.get('M', None)
    rtol = kwargs.get('rtol', RTOL)
    atol = kwargs.get('atol', ATOL)
    maxiter = kwargs.get('maxiter', MAXITER)
    U, info, iters = __solver_core(cg, A, b, x0, M, rtol, atol, maxiter)
    return U, {'info' : info, 'iterations': iters}

def scipy_sparse_gmres(A, b, **kwargs):
    from scipy.sparse.linalg import gmres
    x0 = kwargs.get('x0', None)
    M = kwargs.get('M', None)
    rtol = kwargs.get('rtol', RTOL)
    atol = kwargs.get('atol', ATOL)
    maxiter = kwargs.get('maxiter', MAXITER)
    U, info, iters = __solver_core(gmres, A, b, x0, M, rtol, atol, maxiter)
    return U, {'info' : info, 'iterations': iters}

def scipy_sparse_spsolve(A, b, **kwargs):
    from scipy.sparse.linalg import spsolve
    U = spsolve(A, b)
    return U, {'info' : 'direct solver', 'iterations': 1}
