import importlib
import numpy as np

from .. import probs
from . import subgrad
from .. import utils

importlib.reload(probs)
importlib.reload(subgrad)
importlib.reload(utils)


def dual(
  prob, u, epsilon=None, K=None, x_0=None, log_freq=None, log_prefix=None):
  if x_0 is None:
    x_0 = np.zeros([prob.d])

  if log_prefix is None:
    log_prefix = ''

  def func_wrapper(x):
    return np.sum(u*np.array(prob.funcs(x)))
  def grad_wrapper(x):
    return np.sum(
      np.reshape(u, (prob.m, 1))*np.array(prob.grads(x)),
      axis=0)

  if epsilon is None and K is None:
    epsilon = 1e-5

  dual_prob = probs.FiniteMinimaxProb(
    prob.d, [func_wrapper], [grad_wrapper],
    L=prob.L, sigma=prob.sigma)

  stepsize = (
    1./dual_prob.L if dual_prob.sigma == 0.0 
    else 2./(dual_prob.L+dual_prob.sigma))

  output = subgrad.subGradientDescent_optimize(
      dual_prob, epsilon=epsilon, K=K, x_0=x_0,
      projection_fn=None,
      stepsize_rule='constant', stepsize=stepsize, 
      log_freq=log_freq, log_prefix=log_prefix, log_init=False,
      moreau_env=None)

  return dual_prob.func(output['xk_list'][-1])


def finiteMirrorProx_optimize(
  prob, y_norm=None,
  epsilon=None, K=None, x_0=None, stepsize=None,
  epsilon_check_freq=None, gap_at=None,
  log_freq=None, log_prefix=None, log_init=True):
  """
  Algorithm: Mirror-Prox for min max of finite concave problems

  [Nem]: Nemirovski, 
  PROX-METHOD WITH RATE OF CONVERGENCE O(1/T) FOR VARIATIONAL 
  INEQUALITIES WITH LIPSCHITZ CONTINUOUS MONOTONE OPERATORS 
  AND SMOOTH CONVEX-CONCAVE SADDLE POINT PROBLEMS

  Minimize:
  f(x) = \\min_x \\max_i f_i(x)
       = \\min_x \\max_u g(x, u)
  where g(x, u) = \\sum_i u_i f_i(x)

  h(u) = \\min_x g(x, u)
       = \\min_x \\sum_i u_i f_i(x)

  Solved as a convex-concave smooth-minimax problem

  L-smooth G-Lipschitz f_i ==> g(x, y) is max(G, L)-smooth

  Args:
    prob: FiniteMinimaxProb
    y_norm: None, l1 or l2
    epsilon: None or float
    K: None or int
    x_0: None or np.array([m])
    stepsize: None or float
    epsilon_check_freq: None or float
    gap_at: None or 'avg' or 'last'
    log_freq: None or int
    log_prefix: None or str
    log_init: None or bool

  Returns:
    output: dict

  """
  if x_0 is None:
    x_k = np.zeros([prob.d])
  else:
    x_k = np.array(x_0)

  if epsilon_check_freq is None:
    epsilon_check_freq = 1

  if log_prefix is None:
    log_prefix = ''

  # intializing dual variable
  u_k = np.ones([prob.m])/prob.m

  if y_norm is None:
    y_norm = 'l1'

  if stepsize is None:
    if y_norm == 'l2':
      # Lxx = L - Lipschitz constant of grad_x w.r.t x
      # Lxy = G*sqrt(m) - Lipschitz constant of grad_x w.r.t y
      # Lyx = G*sqrt(m) - Lipschitz constant of grad_y w.r.t x
      # Lyy = 0        - Lipschitz constant of grad_y w.r.t y
      stepsize_x = 1./2/max(prob.G*(prob.m**0.5), prob.L) # L2 norm for y
    elif y_norm == 'l1':
      # Lxx = L - Lipschitz constant of grad_x w.r.t x
      # Lxy = G - Lipschitz constant of grad_x w.r.t y
      # Lyx = G - Lipschitz constant of grad_y w.r.t x
      # Lyy = 0        - Lipschitz constant of grad_y w.r.t y
      stepsize_x = 1./2/max(prob.G, prob.L) # L1 norm for y
  else:
    stepsize_x = stepsize
  stepsize_u = stepsize_x

  epsilon_dual = None if epsilon is None else epsilon**0.5/10
  K_dual = 20 if epsilon_dual is None else None
  log_freq_dual = None if log_freq is None else log_freq*10.
  log_prefix_dual = '{} dual: '.format(log_prefix)
  gap = prob.func(x_k) - dual(
    prob, u_k, epsilon=epsilon_dual, K=K_dual,
    x_0=x_k, log_freq=log_freq_dual, log_prefix=log_prefix_dual)

  xk_list = []
  uk_list = []
  gap_list = []

  xk_list.append(x_k)
  uk_list.append(u_k)
  gap_list.append(gap)

  k = 0
  grad_norm = utils.l2_norm(prob.subgrad(x_k))
  
  if log_init and log_freq is not None:
    log_string = (
      '{}k={}, x_k={}, f(x_k)={}, '
      '||subgrad||={}, u_k={}, f(x_k) - h(u_k)={}'.format(
      log_prefix, k, x_k, prob.func(x_k), grad_norm, u_k, gap))
    print('{}'.format(log_string))

  if K is not None:
    loop_condition = lambda k, gap: k < K
  elif epsilon is not None:
    loop_condition = lambda k, gap: gap >= epsilon
  k = 0

  while loop_condition(k, gap):
    tx_kplus1 = x_k - stepsize_x*np.sum(
      u_k.reshape(prob.m, 1)*
      np.array(prob.grads(x_k)), axis=0)

    grad_uk = np.array(prob.funcs(x_k))
    if y_norm == 'l2':
      # L2 norm update
      tu_kplus1 = utils.project_to_simplex(u_k + stepsize_u*grad_uk)
    elif y_norm == 'l1':
      # KL divergence update
      tu_kplus1 = utils.mirror_descent_simplex(u_k, -grad_uk, stepsize_u)

    x_kplus1 = x_k - stepsize_x*np.sum(
      tu_kplus1.reshape(prob.m, 1)*
      np.array(prob.grads(tx_kplus1)), axis=0 )

    grad_tukplus1 = np.array(prob.funcs(tx_kplus1))
    if y_norm == 'l2':
      # L2 norm update
      u_kplus1 = utils.project_to_simplex(u_k + stepsize_u*grad_tukplus1)
    elif y_norm == 'l1': 
      # KL divergence update
      u_kplus1 = utils.mirror_descent_simplex(u_k, -grad_tukplus1, stepsize_u)

    x_k = x_kplus1
    u_k = u_kplus1

    if (k+1)%epsilon_check_freq == 0:
      xk_list.append(x_k)
      uk_list.append(u_k)

      gap = np.inf
      if gap_at is None or gap_at == 'avg':
        xk_avg = np.mean(xk_list, axis=0)
        uk_avg = np.mean(uk_list, axis=0)
        gap_avg = prob.func(xk_avg) - dual(
          prob, uk_avg, epsilon=epsilon_dual, K=K_dual,
          x_0=xk_avg, log_freq=log_freq_dual, log_prefix=log_prefix_dual)
        if gap_avg <= gap:
          gap = gap_avg
          x_min = xk_avg
          u_min = uk_avg
      if gap_at is None or gap_at == 'last':
        gap_last = prob.func(x_k) - dual(
          prob, u_k, epsilon=epsilon_dual, K=K_dual,
          x_0=x_k, log_freq=log_freq_dual, log_prefix=log_prefix_dual)
        if gap_last <= gap:
          gap = gap_last
          x_min = x_k
          u_min = u_k
      gap_list.append(gap)

    k += 1

    if log_freq is not None and k%log_freq == 0 and k%epsilon_check_freq == 0:
      grad_norm = utils.l2_norm(prob.subgrad(x_k))
      log_string = (
        '{}k={}, x_k={}, f(x_k)={}, '
        '||subgrad||={}, u_k={}, f(x_k) - h(u_k)~={}'.format(
        log_prefix, k, x_k, prob.func(x_k), grad_norm, u_k, gap))
      print('{}'.format(log_string))

  output = {
    'xk_list': xk_list,
    'uk_list': uk_list,
    'gap_list': gap_list,
  }

  return output