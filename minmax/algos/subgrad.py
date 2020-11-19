import importlib
import numpy as np

from .. import utils

importlib.reload(utils)

def get_stepsize_constant(stepsize_rule, stepsize, prob, K):
  if stepsize_rule == 'davis':
    # Theoretical stepsize on pg. 6
    R = min(prob.L*(prob.D**2), prob.D*prob.G)
    stepsize_const = (R/prob.L/(prob.G**2))**0.5
  elif stepsize_rule in ['davis_gamma', 'davis_gamma_var']:
    # On pg. 6
    # assert K is not None
    stepsize_const = stepsize
  elif stepsize_rule == 'subgrad':
    # standard SGD variable stepsize
    stepsize_const = (prob.D/prob.G)
  elif stepsize_rule == 'constant':
    stepsize_const = stepsize

  return stepsize_const


def get_stepsize_at_k(stepsize_rule, stepsize_const, k, K):
  if stepsize_rule in [
      'davis', # Theoretical stepsize on pg. 6
      'davis_gamma_var',  # variable gamma stepsize on pg. 6
      'subgrad', # variable stepsize for standard SGD
    ]:
    _stepsize = stepsize_const/((k+1)**0.5)
  elif stepsize_rule == 'davis_gamma':
    # constant stepsize on pg. 6
    _stepsize = stepsize_const/((K+1)**0.5)
  elif stepsize_rule == 'constant':
    # constant stepsize
    _stepsize = stepsize_const

  return _stepsize


def subGradientDescent_optimize(
  prob, epsilon=None, K=None, x_0=None, 
  projection_fn=None,
  stepsize_rule='davis', stepsize=None,
  best_criteria='function',
  log_freq=None, log_prefix=None, log_init=True,
  moreau_env=None):
  """
  Algorithm 1: Proximal stochastic subgradient method

  [Davis, Drusvyatskiy]
  Stochastic subgradient method converges at the 
  rate O(k^{−1/4}) on weakly convex functions

  Minimize:
  \\min_x \\max_i f_i(x)

  Args:
    prob: FiniteMinimaxProb
    epsilon: None or float
    K: None or int
    x_0: None or np.array([m])
    projection_fn: None or project(x)
    stepsize_rule: 'davis'/'davis_gamma'/'davis_gamma_var'/'subgrad'
    stepsize: None or float,
    best_criteria: 'function' or 'moreau',
    log_freq: None or int
    log_prefix: None or str
    log_init: None or bool
    moreau_env: None or MoreauEnv

  Returns:
    output: dict
  """

  if log_prefix is None:
    log_prefix = ''

  if x_0 is None:
    x_k = np.zeros([prob.d])
  else:
    x_k = np.array(x_0)

  if epsilon is None and K is None:
    K = 10**3

  stepsize_const = get_stepsize_constant(
    stepsize_rule, stepsize, prob, K)

  if projection_fn is None:
    projection_fn = lambda x: x

  grad_norm = utils.l2_norm(prob.subgrad(x_k))

  xk_list = []
  func_list = []
  grad_norm_list = []
  xk_best_list = []
  func_best_list = []
  grad_norm_best_list = []
  if moreau_env is not None:
    moreau_grad_norm_best_list = []
  else:
    moreau_log_string = ''

  xk_list.append(x_k)
  func_list.append(prob.func(x_k))
  grad_norm_list.append(grad_norm)
  xk_best_list.append(x_k)
  func_best_list.append(prob.func(x_k))
  grad_norm_best_list.append(grad_norm)

  if K is not None:
    loop_condition = lambda k, grad_norm: k < K
  elif epsilon is not None:
    loop_condition = lambda k, grad_norm: grad_norm >= epsilon

  k = 0

  if moreau_env is not None:
    moreau_grad = moreau_env.grad(x_k)
    moreau_grad_norm = utils.l2_norm(moreau_grad)
    moreau_grad_norm_best_list.append(moreau_grad_norm)
    moreau_log_string = ', ||moreau_grad||={:.6g}'.format(moreau_grad_norm)

  if log_init and log_freq is not None:
    np.set_printoptions(precision=6)
    log_string = '{}k={}, x_k={}, f(x_k)={:.6g}, ||subgrad||={:.6g}{}'.format(
      log_prefix, k, x_k, prob.func(x_k), grad_norm, moreau_log_string)

    print(log_string)
    np.set_printoptions()

  while loop_condition(k, grad_norm):
    _stepsize = get_stepsize_at_k(stepsize_rule, stepsize_const, k, K)

    x_kplus1 = x_k - _stepsize*prob.subgrad(x_k)
    x_kplus1 = projection_fn(x_kplus1)

    grad_norm = utils.l2_norm(prob.subgrad(x_kplus1))

    x_k = x_kplus1

    xk_list.append(x_k)
    func_list.append(prob.func(xk_list[-1]))
    grad_norm_list.append(grad_norm)
    if best_criteria == 'function':
      if prob.func(x_k) < prob.func(xk_best_list[-1]):
        xk_best_list.append(x_k)
        grad_norm_best_list.append(grad_norm)

        if moreau_env is not None:
          moreau_grad = moreau_env.grad(x_k)
          moreau_grad_norm = utils.l2_norm(moreau_grad)
          moreau_grad_norm_best_list.append(moreau_grad_norm)
          moreau_log_string = ', ||moreau_grad||={:.6g}'.format(moreau_grad_norm)
      else:
        xk_best_list.append(xk_best_list[-1])
        grad_norm_best_list.append(grad_norm_best_list[-1])
        if moreau_env is not None:
          moreau_grad_norm_best_list.append(moreau_grad_norm_best_list[-1])
    elif best_criteria == 'moreau':
      assert moreau_env is not None
      moreau_grad = moreau_env.grad(x_k)
      moreau_grad_norm = utils.l2_norm(moreau_grad)
      
      if moreau_grad_norm < moreau_grad_norm_best_list[-1]:
        moreau_grad_norm_best_list.append(moreau_grad_norm)
        xk_best_list.append(x_k)
        grad_norm_best_list.append(grad_norm)
      else:
        moreau_grad_norm_best_list.append(
          moreau_grad_norm_best_list[-1])
        xk_best_list.append(xk_best_list[-1])
        grad_norm_best_list.append(grad_norm_best_list[-1])

      moreau_log_string = ', ||moreau_grad||={:.6g}'.format(
        moreau_grad_norm_best_list[-1])
    func_best_list.append(prob.func(xk_best_list[-1]))

    k += 1

    if log_freq is not None and k%log_freq == 0:
      np.set_printoptions(precision=6)
      log_string = '{}k={}, x_k={}, f(x_k)={:.6g}, ||subgrad||={:.6g}'.format(
        log_prefix, k, x_k, prob.func(x_k), grad_norm)
      log_string = '{}, x_k_b={}, f(x_k_b)={:.6g}, ||subgrad_b||={:.6g}'.format(
        log_string, xk_best_list[-1], 
        prob.func(xk_best_list[-1]), grad_norm_best_list[-1])
      if moreau_env is not None:
        log_string = '{}, ||moreau_grad_b||={:.6g}'.format(
          log_string, moreau_grad_norm_best_list[-1])
      print(log_string)
      np.set_printoptions()

  output = {
    'xk_list': xk_list,
    'func_list': func_list,
    'grad_norm_list': grad_norm_list,
    'xk_best_list': xk_best_list,
    'func_best_list': func_best_list,
    'grad_norm_best_list': grad_norm_best_list,
  }
  if moreau_env is not None:
    output['moreau_grad_norm_best_list'] = moreau_grad_norm_best_list
  return output