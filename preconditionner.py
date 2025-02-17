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
    from scipy.sparse.linalg import spilu
    fill_factor = kwargs.get('fill_factor', 10) # default value is 10 in scipy
    superLU = spilu(A, fill_factor=fill_factor)
    linear_operator = sps.linalg.LinearOperator(A.shape, superLU.solve)
    return linear_operator