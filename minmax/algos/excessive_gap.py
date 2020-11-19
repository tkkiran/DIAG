import importlib
import numpy as np

from .. import utils

importlib.reload(utils)

def V_excessive_gap(grad, u, L, show_plot=False):
  """
  Solving Dual Objective on Pg. 147 after Lemma 6 of 
  Nesterov, Yu. "Smooth minimization of non-smooth functions."
  Mathematical programming 103.1 (2005): 127-152.

  Args:
    grad: np.array([m])
    u: np.array([m])
    L: float
    show_plot: bool; for visual debugging

  Returns:
    u_min: np.array([m])
  """  
  if len(u) == 1.:
    return np.ones([1])

  grad = np.array(-grad)
  u = np.array(u)
  sort_idx = np.argsort(grad)

  # normalizing, so that sum(grad) = 0
  grad_sort = grad[sort_idx]
  grad_min = grad_sort.min()
  grad_sort = grad_sort - grad_min

  u_sort = u[sort_idx]

  # Dual Objective on Pg. 147 after Lemma 6
  # Nesterov, Yu. "Smooth minimization of non-smooth functions."
  # Mathematical programming 103.1 (2005): 127-152.
  # Notice that we reparameterize tau = 2*tau
  objective = lambda tau: (u_sort*np.maximum(grad_sort - tau, 0)).sum() + (tau**2)/8/L

  # Procedure: Find when lower and upper subgradients 
  # of the dual objective switches signs

  subgrad_lb = np.zeros_like(grad_sort)
  subgrad_ub = np.zeros_like(grad_sort)
  idx = 0
  subgrad_lb[idx] = ( -(u_sort[idx:]).sum() + grad_sort[idx]/4/L )
  for idx in range(len(grad_sort)):
    subgrad_ub[idx] = ( -(u_sort[idx+1:]).sum() + grad_sort[idx]/4/L )

    # min at tau=grad_sort[idx]
    if (np.sign(subgrad_lb[idx]) != np.sign(subgrad_ub[idx]) or 
        np.sign(subgrad_lb[idx]) == 0 or np.sign(subgrad_lb[idx]) == 0):
      idx_min = idx
      tau_min = grad_sort[idx]
      u_sort_min = np.zeros_like(u_sort, dtype=np.float)
      u_sort_min[:idx] = u_sort[:idx]
      u_sort_min[idx] = np.sum(u_sort[idx:]) - tau_min/4/L 
      u_sort_min[0] += (u_sort[idx] - u_sort_min[idx])
      u_sort_min[0] += np.sum(u_sort[idx+1:])

      break

    subgrad_lb[idx+1] = ( -(u_sort[idx+1:]).sum() + grad_sort[idx+1]/4/L )

    # min between tau= grad_sort[idx]={} and tau=grad_sort[idx+1]={}
    if (np.sign(subgrad_ub[idx]) != np.sign(subgrad_lb[idx+1]) or 
        np.sign(subgrad_ub[idx]) == 0 or np.sign(subgrad_lb[idx+1]) == 0):
      idx_min = idx
      # minimizing the effective quadratic between idx and idx+1
      tau_min = 4*L*np.sum(u_sort[idx+1:])
      u_sort_min = np.zeros_like(u_sort, dtype=np.float)
      u_sort_min[:idx+1] = u_sort[:idx+1]
      u_sort_min[0] += np.sum(u_sort[idx+1:])

      break

  dual_value = objective(tau_min)
  primal_value = (np.sum(-grad_sort*(u_sort_min - u_sort)) - 
                  L/2*(np.abs(u_sort_min - u_sort).sum())**2)

  # validating the result
  assert (np.sum(u_sort_min) - np.sum(u_sort)) < 1.e-15
  assert np.abs(tau_min - 0.5*4*L*np.sum(np.abs(u_sort_min-u_sort))) < 1.e-5
  assert np.abs(dual_value - primal_value) < 1.e-14

  # visual debugging
  if show_plot:
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    tau_list = np.arange(grad_sort.min()+1e-16, grad_sort.max()+2e-16, (grad_sort.max()+1e-16)/100.)
    tau_list = np.concatenate([tau_list, grad_sort])
    tau_list = np.sort(tau_list)
    obj_values = np.array([objective(tau) for tau in tau_list])

    ax2.plot(tau_list, obj_values, 'b-')

    vertical_y = np.arange(
      obj_values.min()+1e-16, obj_values.max()+(obj_values.max()-obj_values.min())/3+2e-16,
      (obj_values.max()-obj_values.min()+1e-16)/3)
    for _grad in grad_sort:
      ax1.plot(_grad*np.ones_like(vertical_y), vertical_y, 'y--')
    
    ax1.plot(tau_min*np.ones_like(vertical_y), vertical_y, 'r-.')

    x_step = []
    g_step = []
    for grad_idx, _grad in enumerate(grad_sort):
      x_step.append(4*L*u_sort[grad_idx:].sum())
      g_step.append(_grad)
    ax1.step(g_step, x_step, 'k--')
    ax1.plot(g_step, g_step, 'c.-')

    plt.show()

  u_min = np.zeros_like(u)
  u_min[sort_idx] = u_sort_min
  return u_min


def grad_phi(prob, u):
  """
  Gradient of phi(u) = min_x g(x, u)

  [N]: list item 3 on Pg. 248

  Args:
    prob: AffineQuadraticMinimaxProb
    u: np.array([m])

  Returns:
    grad_phi: np.array([m])      
  """
  # [N]: Pg. 247
  b = prob.ip(prob.g_i, prob.x_i) - prob.f_i
  x = prob.g_i.T.dot(u)/prob.sigma - prob.x_f
  return -( b + prob.g_i.dot(x) )


def grad_phi_2(prob, u):
  """
  Duplicate of def grad_phi(prob, u)
  """
  return(
    prob.f_i + prob.ip(
      prob.g_i, 
      prob.x_f - 
      prob.ip(prob.g_i.T, u)/prob.sigma - prob.x_i))


def V(prob, u, L_phi):
  """
  Applies operator V on u

  [N]: list item 3 on Pg. 248 and Eq. (7.6) on Pg. 245 and Eq. (7.1) on Pg. 244

  Args:
    prob: AffineQuadraticMinimaxProb
    u: np.array([m])

  Returns:
    V: np.array([m]) such that V.sum() = 1  
  """
  return V_excessive_gap(
    grad_phi(prob, u), u, L_phi, show_plot=False)


def u_mu(prob, x, mu):
  """
  Maximizer of g(x, u) - mu*d(u) over u

  [N]: list item 1 on Pg. 248 and Eq. (2.5) on Pg. 237

  Args:
    prob: AffineQuadraticMinimaxProb
    x: np.array([d])
    mu: real number

  Returns:
    u_mu: np.array([m])
  """
  s = (prob.f_i + 
       prob.ip(prob.g_i, x - prob.x_i) )/mu
  
  # Overflow safe exponential normalization
  # return np.exp(s)/np.exp(s).sum()
  return utils.exp_normalize(s)


def affineQuadraticExcessiveGap_optimize(
  prob, K=None, epsilon=None, log_freq=None, log_prefix=None,
  log_init=True):
  """
  Excessive Gap Technique

  Method 2 on Pg. 247:
  [N] Nesterov, Yu. \"Excessive gap technique in nonsmooth convex minimization.\"
  SIAM Journal on Optimization 16.1 (2005): 235-249.

  Minimize:
  0.5 sigma ||x - x_f||^2 + \\max_i [f_i + <g_i, x - x_i>]
  
  Args:
    prob: AffineQuadraticMinimaxProb
    K: Nof. iterations. None or integer
    epsilon: Duality gap tolerance. None or real number
    log_freq: Frequency of logging. None (no logs) or Integer.

  Return:
    xk_bar_list: List of primal solutions over iterations.
    uk_bar_list: List of dual solutions over iterations.
    gap_list:  List of duality gap over iterations.
  """

  # [N]: Eqn. (7.2) on Pg. 244, and ||A||_{1, 2} on Pg. 248
  L_phi = (
    np.max([utils.l2_norm(g) for g in prob.g_i])**2)/prob.sigma
  L_phi_2 = np.max(
    prob.ip(prob.g_i, prob.g_i))/prob.sigma

  if K is None and epsilon is None:
    raise ValueError('Either K or epsilon should be given!')

  if log_prefix is None:
    log_prefix = ''

  # Method 2 on Pg. 247 (Step 1: Initialization)
  mu_k = 2.0*L_phi # sigma_2 = 1. ([N] end of Pg. 247)

  # [N] Pg. 237 and 247-248: u0 = argmin d2(u)
  uk = np.ones([prob.m])/prob.m

  xk_bar_list = []
  uk_bar_list = []
  gap_list = []

  # Method 2 on Pg. 247 (Step 1: Initialization)
  xk_bar = prob.x0(uk)
  uk_bar = V(prob, uk, L_phi)
  gap = prob.f(xk_bar) - prob.phi(uk_bar)

  xk_bar_list.append(xk_bar)
  uk_bar_list.append(uk_bar)
  gap_list.append(gap)

  if K is not None:
    loop_condition = lambda k, gap: k < K
  elif epsilon is not None:
    loop_condition = lambda k, gap: gap >= epsilon

  k = 0
  if log_init and log_freq is not None:
    print('{}k={}, x_k={}, u_k={}, gap={}'.format(
      log_prefix, k, xk_bar, uk_bar, gap))

  while loop_condition(k, gap):
    tau_k = 2.0/(k+3.0)
    uk_hat = (1.0-tau_k)*uk_bar + tau_k*u_mu(prob, xk_bar, mu_k)

    mu_k = (1.0-tau_k)*mu_k
    xk_bar = (1.0-tau_k)*xk_bar + tau_k*prob.x0(uk_hat)
    uk_bar = V(prob, uk_hat, L_phi)

    xk_bar_list.append(xk_bar)
    uk_bar_list.append(uk_bar)

    k += 1
    gap = prob.f(xk_bar) - prob.phi(uk_bar)
    gap_list.append(gap)

    if log_freq is not None and k%log_freq == 0:
      np.set_printoptions(suppress=True, precision=3)
      print('{}k={}, x_k={}, u_k={}, gap={}'.format(
        log_prefix, k, xk_bar, uk_bar, gap))
      print('\ttau_k={}, uk_hat={}, mu_k={}'.format(
        tau_k, uk_hat, mu_k))
      np.set_printoptions()


  if log_freq is not None:
    print('{}k={}, x_k={}, u_k={}, gap={}'.format(
      log_prefix, k, xk_bar, uk_bar, gap))

  output = {
    'xk_bar_list': xk_bar_list,
    'uk_bar_list': uk_bar_list,
    'gap_list': gap_list,
  }
  return output
