import importlib
import numpy as np

from .. import utils
from .. import probs
from . import excessive_gap as excess_gap

importlib.reload(utils)
importlib.reload(probs)
importlib.reload(excess_gap)

def proxFDIAG_optimize(
  prob, epsilon, x_0=None, log_freq=None, log_prefix=None,
  excess_gap_log_freq=10000,
  moreau_env=None, moreau_stop=None, moreau_stop_epsilon=None):
  """
  Algorithm 5: Proximal Finite Dual Implicit Accelerated 
  Gradient (Prox-FDIAG) for finite 
  nonconvex-concave minimax optimization
  [TJNO]: KK Thekumparampil, P Jain, P Netrapalli, S Oh,
  \"Efficient algorithms for smooth minimax optimization\"

  Minimize:
  \\min_x \\max_i f_i(x)

  Args:
    prob: FiniteMinimaxProb
    epsilon: float
    x_0: None or np.array([m])
    log_freq: None or int
    log_prefix: None or str
    excess_gap_log_freq: int
    moreau_env: None or MoreauEnv
    moreau_stop: None or bool
    moreau_stop_epsilon: None or float

  Returns:
    output: dict
  """
  if x_0 is None:
    x_k = np.zeros([prob.d])
  else:
    x_k = x_0

  if log_prefix is None:
    log_prefix = ''

  if moreau_stop is None:
    moreau_stop = False
    if moreau_stop_epsilon is None:
      moreau_stop_epsilon = epsilon

  tepsilon = (epsilon**2)/64.0/prob.L # epsilon-tilde [Algorithm 5, TJNO]
  gap = tepsilon

  xk_list = []
  func_list = []
  xk_inner_list = []
  nof_grads_list = []
  gap_list = []
  if moreau_env is not None:
    moreau_grad_norm_list = []
  else:
    moreau_log_string = ''

  xk_list.append(x_k)
  func_list.append(prob.func(x_k))
  xk_inner_list.append(x_k)
  nof_grads_list.append(0)
  gap_list.append(gap)

  k = 0
  grad_norm = utils.l2_norm(prob.subgrad(x_k))

  if moreau_env is not None:
    moreau_grad = moreau_env.grad(x_k)
    moreau_grad_norm = utils.l2_norm(moreau_grad)
    moreau_grad_norm_list.append(moreau_grad_norm)
    moreau_log_string = ', ||moreau_grad||={:.6g}'.format(moreau_grad_norm)
  
  if log_freq is not None:
    log_string = '{}k=0, x_k={}, f(x_k)={}, ||subgrad||={}{}'.format(
      log_prefix, x_k, prob.func(x_k), grad_norm, moreau_log_string)
    print(log_string)

  while 3.0*tepsilon/4.0 < gap:
    quad_approx = probs.AffineQuadraticMinimaxProb(
      g_i=prob.grads(x_k), 
      f_i=prob.funcs(x_k), 
      x_i=x_k, x_f=x_k, sigma=prob.L)

    xk_excess_gap_list = excess_gap.affineQuadraticExcessiveGap_optimize(
      prob=quad_approx, epsilon=tepsilon/4.0, 
      log_freq=excess_gap_log_freq, log_prefix='\t _',
      log_init=False)['xk_bar_list']
    
    x_kplus1 = xk_excess_gap_list[-1]
    xk_inner_list.extend(xk_excess_gap_list)
    nof_grads_list.append(nof_grads_list[-1]+len(xk_excess_gap_list))

    gap = prob.func(x_k) - quad_approx.f(x_kplus1)

    x_k = x_kplus1
    xk_list.append(x_k)
    func_list.append(prob.func(x_k))
    gap_list.append(gap)

    k += 1

    grad_norm = utils.l2_norm(prob.subgrad(x_k))

    if moreau_env is not None:
      moreau_grad = moreau_env.grad(x_k)
      moreau_grad_norm = utils.l2_norm(moreau_grad)
      moreau_grad_norm_list.append(moreau_grad_norm)
      moreau_log_string = ', ||moreau_grad||={:.6g}'.format(moreau_grad_norm)

    if log_freq is not None and k%log_freq == 0:
      log_string = (
        '{}k={}, x_k={}, f(x_k)={}, '
        '||subgrad||={}, f(x_k-1) - f(x_k, x_k-1)={}{}'.format(
        log_prefix, k, x_k, prob.func(x_k), grad_norm, gap, moreau_log_string))
      print(log_string)

    if moreau_env is not None and moreau_stop and moreau_grad_norm <= moreau_stop_epsilon:
      break

  output = {
    'xk_list': xk_list,
    'func_list': func_list,
    'gap_list': gap_list,
    'nof_grads_list': nof_grads_list,
    'xk_inner_list': xk_inner_list,
  }
  if moreau_env is not None:
    output['moreau_grad_norm_list'] = moreau_grad_norm_list
  return output


def adaptiveProxFDIAG_optimize(
  prob, epsilon, epsilon_start=1., epsilon_division=2.,
  x_0=None, log_freq=None, log_prefix=None,
  excess_gap_log_freq=10000,
  moreau_env=None, moreau_stop=None):
  """
  Algorithm 6: Adaptive Proximal Finite Dual Implicit Accelerated 
  Gradient (Prox-FDIAG) for finite 
  nonconvex-concave minimax optimization
  
  [TJNO]: KK Thekumparampil, P Jain, P Netrapalli, S Oh,
  \"Efficient algorithms for smooth minimax optimization\"

  Minimize:
  \\min_x \\max_i f_i(x)

  Args:
    prob: FiniteMinimaxProb
    epsilon: float
    epsilon_start: float
    epsilon_division: float
    x_0: None or np.array([m])
    log_freq: None or int
    log_prefix: None or str
    excess_gap_log_freq: int
    moreau_env: None or MoreauEnv
    moreau_stop: None or bool

  Returns:
    output: dict
  """
  if moreau_stop is None:
    moreau_stop = True
  outputs = []
  epsilon_curr = max(epsilon, epsilon_start)
  x_k = x_0
  init_iter = True
  while True:
    print('epsilon_curr', epsilon_curr)

    _output = proxFDIAG_optimize(
      prob, epsilon_curr, x_0=x_k, 
      log_freq=log_freq, log_prefix=log_prefix,
      excess_gap_log_freq=excess_gap_log_freq,
      moreau_env=moreau_env, 
      moreau_stop=moreau_stop, moreau_stop_epsilon=epsilon)
    x_k = _output['xk_list'][-1]
    if not init_iter:
      _output['nof_grads_list'] = (
        np.array(_output['nof_grads_list']) +
        outputs[-1]['nof_grads_list'][-1]
      )
    outputs.append(_output)

    if (moreau_env is not None and moreau_stop and 
        _output['moreau_grad_norm_list'][-1] <= epsilon):
      break
    if np.isclose(epsilon_curr, epsilon, atol=0.0):
      break

    init_iter = False
    epsilon_curr = max(epsilon, epsilon_curr/epsilon_division)

  output = {}
  for key in outputs[0]:
    output[key] = np.concatenate([
      _output[key] for _output in outputs]) 

  return output