import importlib
import numpy as np

from .. import probs
from . import mirror_prox
from . import subgrad
from .. import utils

importlib.reload(probs)
importlib.reload(mirror_prox)
importlib.reload(subgrad)
importlib.reload(utils)


def finiteAccelPrimalDual_optimize(
  prob, epsilon=None, K=None, x_0=None, stepsize=None,
  epsilon_check_freq=None, gap_at=None,
  log_freq=None, log_prefix=None, log_init=True):
  """
  Algorithm: Accelerated Primal-Dual method

  [HA]: Hamedani, Aybat
  "A Primal-Dual Algorithm for General Convex-Concave 
  Saddle Point Problemss"

  Minimize:
  f(x) = \\min_x \\max_i f_i(x)
       = \\min_x \\max_u g(x, u)
  where g(x, u) = \\sum_i u_i f_i(x)

  h(u) = \\min_x g(x, u)
       = \\min_x \\sum_i u_i f_i(x)

  Solved as a convex-concave smooth-minimax problem

  L-smooth G-Lipschitz f_i ==> g(x, y) is max(G, L)-smooth
  Lyy = 0 for g (linear in y)

  Args:
    prob: FiniteMinimaxProb
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

  if stepsize is None:
    # Thm 2.1 part II
    Lxx = prob.L # Lipschitz constant of grad_x w.r.t x
    Lxy = prob.G # Lipschitz constant of grad_x w.r.t y
    Lyx = prob.G # Lipschitz constant of grad_y w.r.t x
    Lyy = 0      # Lipschitz constant of grad_y w.r.t y
    alpha = Lyx # 1.
    c_tau = 1.
    c_sigma = 0.5
    stepsize_x = c_tau/(Lxx + (Lyx**2)/alpha)
    stepsize_u = c_sigma/alpha
  else:
    stepsize_x = stepsize
    stepsize_u = stepsize

  theta_k = 1
  xk_avg = x_k
  uk_avg = u_k
  weight_sum = 0

  epsilon_dual = None if epsilon is None else epsilon**0.5/10
  K_dual = 20 if epsilon_dual is None else None
  log_freq_dual = None if log_freq is None else log_freq*10.
  log_prefix_dual = '{} dual: '.format(log_prefix)
  gap = prob.func(x_k) - mirror_prox.dual(
    prob, u_k, epsilon=epsilon_dual, K=K_dual,
    x_0=x_k, log_freq=log_freq_dual, log_prefix=log_prefix_dual)

  xk_list = []
  xk_avg_list = []
  uk_list = []
  uk_avg_list = []
  gap_list = []

  xk_list.append(x_k)
  xk_avg_list.append(x_k)
  uk_list.append(u_k)
  uk_avg_list.append(u_k)
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
  
  x_kminus1, u_kminus1 = x_k, u_k
  while loop_condition(k, gap):
    grad_k_y = np.array(prob.funcs(x_k))
    grad_kminus1_y = np.array(prob.funcs(x_kminus1))

    s_k = (1+theta_k)*grad_k_y - theta_k*grad_kminus1_y
    u_kplus1 = utils.mirror_descent_simplex(u_k, -s_k, stepsize_u)

    grad_k_x = np.sum(
      u_kplus1.reshape(prob.m, 1)*
      np.array(prob.grads(x_k)), axis=0 )
    x_kplus1 = x_k - stepsize_x*grad_k_x

    x_kminus1, u_kminus1 = x_k, u_k
    x_k = x_kplus1
    u_k = u_kplus1

    weight_sum = weight_sum + stepsize_u
    weight = stepsize_u/weight_sum
    xk_avg = (1-weight)*xk_avg + weight*x_k
    uk_avg = (1-weight)*uk_avg + weight*u_k

    if (k+1)%epsilon_check_freq == 0:
      gap = np.inf
      if gap_at is None or gap_at == 'avg':
        gap_avg = prob.func(xk_avg) - mirror_prox.dual(
          prob, uk_avg, epsilon=epsilon_dual, K=K_dual,
          x_0=xk_avg, log_freq=log_freq_dual, log_prefix=log_prefix_dual)
        if gap_avg <= gap:
          gap = gap_avg
          x_min = xk_avg
          u_min = uk_avg
      if gap_at is None or gap_at == 'last':
        gap_last = prob.func(x_k) - mirror_prox.dual(
          prob, u_k, epsilon=epsilon_dual, K=K_dual,
          x_0=x_k, log_freq=log_freq_dual, log_prefix=log_prefix_dual)
        if gap_last <= gap:
          gap = gap_last
          x_min = x_k
          u_min = u_k
      gap_list.append(gap)

      xk_list.append(x_k)
      xk_avg_list.append(xk_avg)
      uk_list.append(u_k)
      uk_avg_list.append(uk_avg)
      gap_list.append(gap)

    theta_k = 1/(1+prob.sigma*stepsize_x)**0.5
    stepsize_x = theta_k*stepsize_x
    stepsize_u = stepsize_u/theta_k

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
    'xk_avg_list': xk_avg_list,
    'uk_list': uk_list,
    'uk_avg_list': uk_avg_list,
    'gap_list': gap_list,
  }

  return output