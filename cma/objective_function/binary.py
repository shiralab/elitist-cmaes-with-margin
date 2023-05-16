#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ..objective_function.base import *
import numpy as np


# public symbols
__all__ = ['OneMax', 'LeadingOnes', 'BinVal']

def initial_setting_for_gaussian(func_instance, random=True):
    """
    Return random initial vector within the range or constant initial vector.

    :type func_instance: object
    :type random: bool
    :return: initial mean vector
    :rtype: array_like, shape=(d), dtype=float
    :return: initial sigma
    :rtype: float
    """
    a, b = 1., 3.
    # mean vector:
    #   continuous and integer: sample from uniform distribution [a, b]
    #   binary: 0.5
    # sigma: (b - a) / 2
    return np.full(func_instance.d, 0.5), (b - a) / 2

class OneMax(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, d, target_eval=None, max_eval=1e4):
        self.d = d
        if target_eval is None:
            target_eval = self.d
        super(OneMax, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = X.sum(axis=1)
        self._update_best_eval(evals)
        return evals

class BinVal(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, d, target_eval=None, max_eval=1e4):
        self.d = d
        self.coeff = np.array([2**i for i in range(d)], dtype=object)
        if target_eval is None:
            target_eval = np.sum(self.coeff)
        super(BinVal, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        X = X.astype(int)
        evals = np.sum(self.coeff * X, axis=1, dtype=object)
        self._update_best(evals, X)
        return evals

    def verbose_display(self):
        return ' EvalCount: %d' % self.eval_count + ' BestEval: %e' % self.best_eval + ' NumOf1: %d' % np.sum(self.best_x)

    def _update_best(self, evals, X):
        b = np.argmax(evals)

        if evals[b] > self.best_eval:
            self.best_eval = evals[b]
            self.best_x = X[b]
    
    def clear(self):
        self.eval_count = 0
        self.best_eval = np.inf if self.minimization_problem else -np.inf
        self.best_x = None


class LeadingOnes(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, d, target_eval=None, max_eval=1e4):
        self.d = d
        if target_eval is None:
            target_eval = self.d
        super(LeadingOnes, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = X.argmin(axis=1) + X.prod(axis=1) * self.d
        self._update_best_eval(evals)
        return evals

