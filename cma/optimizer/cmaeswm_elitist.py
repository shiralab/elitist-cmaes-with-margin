#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import gmean
import sys

from ..optimizer.base_optimizer import BaseOptimizer
from ..util.model import GaussianSigmaACA

# public symbols
__all__ = ['CMAESwM_elitist']

class CMAESwM_elitist(BaseOptimizer):
    def __init__(
            self, 
            d,                          # num. of dim.
            discrete_space,             # domain of discrete space
            sampler,                    # sampler for generating candidate solution
            m=None, C=None, sigma=1.,   # initial distribution parameters
            minimal_eigenval=1e-30,     # lower bound of eigenvalues of distribution's covariance matrix
            c_cov=None, c_p=None, damping=None,   # hyperparameter setting of elite preserving CMA-ES
            margin=None,                # setting for margin parameter
            normalize='None',           # 
            min_problem=True,           # True if the optimization problem is minimization problem
            postprocess=False,         # Apply the post-process for integer opt. if this is True (and binary_mode is False)
            tie_success=True,           # consider the tie cases are successful update if True (important for integer/binary opt.)
            enc_m=True                  # Apply discretization of mean vector
        ):
        
        self.model = GaussianSigmaACA(d, m=m, C=C, sigma=sigma, z_space=discrete_space, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.model_init = GaussianSigmaACA(d, m=m, C=C, sigma=sigma, z_space=discrete_space, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.weight_func = None
        self.sampler = sampler
        self.d = d
        self.zd = len(discrete_space)

        # misc parameters
        self.is_better = (lambda x, y: x < y) if min_problem else (lambda x, y: x > y)
        self.best_X = None
        self.best_eval = None

        # SSA parameters
        self.model.sigma = 1. if sigma is None else sigma
        self.damping = 1. + d / 2. if damping is None else damping
        self.c_p = 1. / 12. if c_p is None else c_p
        self.p_target = 2. / 11. 
        self.p_succ = self.p_target

        # CMA parameters
        self.c_cov = 2. / (d ** 2 + 6.) if c_cov is None else c_cov
        self.c_c = 2. / (2. + d)
        self.p_thresh = 0.44
        self.pc = np.zeros((1, d))

        # vars for SSA
        self.p_succ = self.p_target

        # evolution path
        self.pc = np.zeros(d)
        self.gen_count = 0

        # margin parameter (alpha in the paper)
        self.margin = margin if margin is not None else 1. / d
        self.postprocess = postprocess

        self.tie_success = tie_success
        self.enc_m = enc_m

    def sampling_model(self):
        return self.model
    
    def update_step_size(self):
        self.model.sigma = self.model.sigma * np.exp((self.p_succ - self.p_target) / (1. - self.p_target) / self.damping)

    def update(self, X, evals):
        self.gen_count += 1

        # first update
        if self.best_X is None:
            self.best_X = X[0]
            self.best_eval = evals[0]
            self.model.m = X[0]
            return

        lam_succ = self.is_better(evals[0], self.best_eval)
        eq_lam_succ = evals[0] is self.best_eval 
        lam_succ = lam_succ or (eq_lam_succ and self.tie_success)

        self.p_succ = (1. - self.c_p) * self.p_succ + self.c_p * lam_succ
        
        if lam_succ:
            self.update_cov((X[0] - self.model.m) / self.model.sigma)
            self.best_X = X[0]
            self.best_eval = evals[0]
            if self.enc_m:
                self.model.m = self.model.encoding(1, X)[0]
            else:
                self.model.m = (X[0] - self.model.m) * self.model.A + self.model.m
        
        self.update_step_size()

        self.modify_margin()

        if self.postprocess:
            if np.all(self.model.A > 1.):
                min_A = np.min(self.model.A)
                self.model.A /= min_A
                self.model.sigma *= min_A

    def update_cov(self, y):
        if self.p_succ < self.p_thresh:
            self.pc = (1. - self.c_c) * self.pc + (self.c_c * (2. - self.c_c)) ** 0.5 * y
            self.model.C = (1. - self.c_cov) * self.model.C + self.c_cov * (np.outer(self.pc, self.pc))
        else:
            self.pc = (1. - self.c_c) * self.pc
            self.model.C = (1. - self.c_cov) * self.model.C + self.c_cov * ((np.outer(self.pc, self.pc)) + self.c_c * (2. - self.c_c) * self.model.C)

    def modify_margin(self):
        # margin correction (if self.margin = 0, this behaves as CMA-ES)
        if self.margin > 0.:
            num_cont = self.model.d - self.model.zd # = N_continuous
            updated_m_integer = self.model.m[num_cont:, np.newaxis]
            # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
            self.z_lim_low = np.concatenate([self.model.z_lim.min(axis=1).reshape([self.model.zd,1]), self.model.z_lim], 1)
            self.z_lim_up = np.concatenate([self.model.z_lim, self.model.z_lim.max(axis=1).reshape([self.model.zd,1])], 1)
            self.m_z_lim_low = (self.z_lim_low * np.where(np.sort(np.concatenate([self.model.z_lim, updated_m_integer], 1))==updated_m_integer, 1, 0)).sum(axis=1)
            self.m_z_lim_up = (self.z_lim_up * np.where(np.sort(np.concatenate([self.model.z_lim, updated_m_integer], 1))==updated_m_integer, 1, 0)).sum(axis=1)

            # calculate probability low_cdf := Pr(X <= m_z_lim_low) and up_cdf := Pr(m_z_lim_up < X)
            sig_z_sq_Cdiag = self.model.sigma * self.model.A * np.sqrt(np.diag(self.model.C))
            z_scale = sig_z_sq_Cdiag[num_cont:]
            updated_m_integer = updated_m_integer.flatten()
            low_cdf = norm.cdf(self.m_z_lim_low, loc = updated_m_integer, scale = z_scale)
            up_cdf = 1. - norm.cdf(self.m_z_lim_up, loc = updated_m_integer, scale = z_scale)
            mid_cdf = 1. - (low_cdf + up_cdf)
            # edge case
            edge_mask = (np.maximum(low_cdf, up_cdf) > 0.5)
            # otherwise
            side_mask = (np.maximum(low_cdf, up_cdf) <= 0.5)

            C_diag_sq = np.sqrt(np.diag(self.model.C))[num_cont:]
       
            if np.any(edge_mask):
                modify_mask = (np.minimum(low_cdf, up_cdf) < self.margin)
                modify_sign = np.sign(self.model.m[num_cont:] - self.m_z_lim_up)
                
                m_edge_dist = np.maximum(
                    (self.model.m[num_cont:] - self.m_z_lim_low) * (modify_sign == 1), 
                    (self.m_z_lim_up - self.model.m[num_cont:]) * (modify_sign == -1), 
                )

                self.model.A[num_cont:] = \
                    self.model.A[num_cont:] + edge_mask * ( 
                        m_edge_dist / (
                            np.sqrt(chi2.ppf(q = 1.-2*self.margin, df = 1)) * \
                            self.model.sigma * C_diag_sq
                        ) - self.model.A[num_cont:] )

            # correct probability
            low_cdf = np.maximum(low_cdf, self.margin/2.)
            up_cdf = np.maximum(up_cdf, self.margin/2.)
            modified_low_cdf = low_cdf + (1. - low_cdf - up_cdf - mid_cdf) * (low_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3. * self.margin / 2)
            modified_up_cdf = up_cdf + (1. - low_cdf - up_cdf - mid_cdf) * (up_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3. * self.margin / 2)
            modified_low_cdf = np.clip(modified_low_cdf, 1e-10, 0.5 - 1e-10)
            modified_up_cdf = np.clip(modified_up_cdf, 1e-10, 0.5 - 1e-10)
        
            # modify mean vector and A (with sigma and C fixed)
            chi_low_sq = np.sqrt(chi2.ppf(q = 1.-2*modified_low_cdf, df = 1))
            chi_up_sq = np.sqrt(chi2.ppf(q = 1.-2*modified_up_cdf, df = 1))
            
            # simultaneous equations
            # (updated_m_integer) - self.m_z_lim_low = chi_low_sq * self.model.sigma * (self.model.A) * C_diag_sq
            # self.m_z_lim_up - (updated_m_integer) = chi_up_sq * self.model.sigma * (self.model.A) * C_diag_sq
            self.model.A[num_cont:] = self.model.A[num_cont:] + side_mask * ( (self.m_z_lim_up - self.m_z_lim_low) / ((chi_low_sq + chi_up_sq) * self.model.sigma * C_diag_sq) - self.model.A[num_cont:] )
            self.model.m[num_cont:] = self.model.m[num_cont:] + side_mask * ( (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq) / (chi_low_sq + chi_up_sq) - self.model.m[num_cont:] )

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display() 

    def log_header(self):
        return self.model.log_header() + ['sigma'] + ['A%d' % i for i in range(self.d)] + ['psucc']

    def log(self):
        return self.model.log() + ['%e' % self.model.sigma] + ['%e' % i for i in self.model.A] + ['%e' % self.p_succ]



