"""
Linear Algebra Solver for Analytical Problems
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import linalg

class LinearAlgebraSolver:
    """Solve linear algebra problems analytically"""

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve Ax = b

        Returns:
            x: Solution vector
            info: Dictionary with solution information
        """
        try:
            x = linalg.solve(A, b)
            residual = np.linalg.norm(A @ x - b)
            condition_number = np.linalg.cond(A)

            info = {
                'success': True,
                'method': 'direct',
                'residual': residual,
                'condition_number': condition_number,
                'well_conditioned': condition_number < 1e10
            }

            return x, info

        except linalg.LinAlgError as e:
            return None, {'success': False, 'error': str(e)}

    @staticmethod
    def eigenvalue_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Compute eigenvalues and eigenvectors

        Returns:
            eigenvalues: Array of eigenvalues
            eigenvectors: Matrix of eigenvectors
            info: Information dictionary
        """
        try:
            eigenvalues, eigenvectors = linalg.eig(A)

            info = {
                'success': True,
                'num_eigenvalues': len(eigenvalues),
                'max_eigenvalue': np.max(np.abs(eigenvalues)),
                'min_eigenvalue': np.min(np.abs(eigenvalues)),
                'is_symmetric': np.allclose(A, A.T)
            }

            return eigenvalues, eigenvectors, info

        except Exception as e:
            return None, None, {'success': False, 'error': str(e)}

    @staticmethod
    def svd_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Singular Value Decomposition

        Returns:
            U, s, Vh: SVD components
            info: Information dictionary
        """
        try:
            U, s, Vh = linalg.svd(A)

            info = {
                'success': True,
                'rank': np.sum(s > 1e-10),
                'condition_number': s[0] / s[-1] if s[-1] > 0 else np.inf,
                'max_singular_value': s[0],
                'min_singular_value': s[-1]
            }

            return U, s, Vh, info

        except Exception as e:
            return None, None, None, {'success': False, 'error': str(e)}

    @staticmethod
    def least_squares(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Solve least squares problem: min ||Ax - b||^2

        Returns:
            x: Solution vector
            info: Information dictionary
        """
        try:
            x, residuals, rank, s = linalg.lstsq(A, b)

            info = {
                'success': True,
                'rank': rank,
                'residual_norm': np.sqrt(residuals[0]) if len(residuals) > 0 else 0,
                'condition_number': s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf
            }

            return x, info

        except Exception as e:
            return None, {'success': False, 'error': str(e)}
