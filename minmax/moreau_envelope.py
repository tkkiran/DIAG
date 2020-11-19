import importlib
import numpy as np

from . import algos
from . import probs
from . import utils

importlib.reload(algos)
importlib.reload(probs)
importlib.reload(utils)

class FiniteMaxMoreauEnv(object):
  def __init__(
    self, prob, rho=None, epsilon=None, K=None, 
    optim_algo=None, y_norm=None, 
    stepsize=None, epsilon_check_freq=None,
    log_freq=None, log_prefix=None):
    self.prob = prob

    if rho is None:
      rho = 2*prob.L
    _L = prob.L + rho
    _sigma = -prob.L + rho
    assert prob.L < rho

    self.rho = rho
    self._L = _L
    self._sigma = _sigma

    if epsilon is None and K is None:
      epsilon = 1e-8
    self.epsilon = epsilon
    self.K = K

    if optim_algo is None:
      optim_algo = 'FiniteMirrorProx'
    self.optim_algo = optim_algo
    self.y_norm = y_norm
    self.stepsize = stepsize
    self.epsilon_check_freq = epsilon_check_freq

    self.log_freq = log_freq
    if log_prefix is None:
      log_prefix = 'moreau: '
    self.log_prefix = log_prefix

  def create_moreau_prox_prob(self, x):
    prob = self.prob
    rho = self.rho
    x = np.array(x)

    phis = []
    phi_grads = []

    def moreau_wrapper(f, x, rho):
      return lambda _x: f(_x) + (rho/2.)*(utils.l2_norm(_x-x)**2)
    def moreau_grad_wrapper(g, x, rho):
      return lambda _x: g(_x) + (rho)*(_x-x)

    for f_idx in range(len(prob._funcs)):
      phis.append(moreau_wrapper(prob._funcs[f_idx], x, rho))
      phi_grads.append(moreau_grad_wrapper(prob._grads[f_idx], x, rho))

    moreau_prob = probs.FiniteMinimaxProb(
      prob.d, phis, phi_grads, L=self._L, G=prob.G, D=prob.D,
      sigma=self._sigma)
    return moreau_prob

  def minimizer(self, x, return_prob=False):
    x = np.array(x)
    moreau_prob = self.create_moreau_prox_prob(x)

    if self.optim_algo == 'FiniteMirrorProx':
      output = algos.finiteMirrorProx_optimize(
        moreau_prob, epsilon=self.epsilon, K=self.K, x_0=x,
        # stepsize=1./2/max(moreau_prob.G, moreau_prob.L)*(moreau_prob.m**0.5), # faster
        stepsize=self.stepsize,
        epsilon_check_freq=self.epsilon_check_freq, 
        gap_at='last',
        y_norm=self.y_norm,
        log_freq=self.log_freq, log_prefix=self.log_prefix,
        log_init=False)
      x_min_last = output['xk_list'][-1]
      x_min_avg = np.mean(output['xk_list'], axis=0)

    elif self.optim_algo == 'FiniteAPD':
      output = algos.finiteAccelPrimalDual_optimize(
        moreau_prob, epsilon=self.epsilon, K=self.K, x_0=x,
        stepsize=self.stepsize,
        epsilon_check_freq=self.epsilon_check_freq,
        gap_at='last',
        log_freq=self.log_freq, log_prefix=self.log_prefix,
        log_init=False)
      x_min_last = output['xk_list'][-1]
      x_min_avg = output['xk_avg_list'][-1]

    if moreau_prob.func(x_min_avg) <= moreau_prob.func(x_min_last):
      x_min = x_min_avg
    else:
      x_min = x_min_last

    if return_prob:
      return x_min, p
    else:  
      return x_min

  def grad(self, x):
    x_min = self.minimizer(x, return_prob=False)
    return (self.rho)*(x - x_min)

  def func(self, x):
    x_min, p = self.minimizer(x, return_prob=True)
    return p.func(x_min)