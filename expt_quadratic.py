import importlib
from datetime import datetime
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from minmax import algos, probs, utils
from minmax import moreau_envelope as morenv

_parent_log_dir = 'results3'

_script_file = 'expt_quadratic.py'
_script_files = [
  'minmax/algos/proxfdiag.py', 
  'minmax/algos/subgrad.py', 
  'minmax/algos/excessive_gap.py',
  'minmax/algos/mirror_prox.py', 
  'minmax/algos/__init__.py',
  'minmax/moreau_envelope.py',
  'minmax/probs.py',
  'minmax/utils.py',
  ]
temp_log_dir = os.path.join(
  'results/temp', datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(temp_log_dir):
  os.makedirs(temp_log_dir)
temp_script_dir = utils.dump_script(temp_log_dir, _script_file, file_list=_script_files)


def generate_max_quadratic_prob(d, m, x_0, seed):
  x_min, x_max = -3, 3
  c_min, c_max = 1, 5

  funcs = []
  grads = []
  if seed is not None:
    print('random_seed', seed)
    np.random.seed(seed)
  for i in range(m):
    x = []
    for j in range(d):
      x.append(x_min + (x_max - x_min)*np.random.sample())
    c = c_min + (c_max - c_min)*np.random.sample()
    print('i={}, x={}, c={}'.format(i, x, c))

    f, g = utils.generate_qudratic_fn(-1., x, c)
    funcs.append(f)
    grads.append(g)

  fq, gq = utils.generate_qudratic_fn(0.5, 0.0*np.zeros([d]))
  funcs.append(fq)
  grads.append(gq)

  L = 1
  D = 100
  G = 2*L*utils.l2_norm(x_0)

  prob = probs.FiniteMinimaxProb(d, funcs, grads, L, D=D, G=G)
  return prob


def generate_moreau_envelopes(
  prob, epsilon=None, K=None, optim_algo=None,
  y_norm=None, epsilon_check_freq=None,
  log_freq=None):
  if optim_algo is None:
    optim_algo = 'FiniteMirrorProx'
  if y_norm is None:
    y_norm = 'l2'
  if epsilon is None:
    epsilon = 1.e-13
  if epsilon_check_freq is None:
    epsilon_check_freq = 1000
  if log_freq is None:
    log_freq = 1000

  moreau_env = morenv.FiniteMaxMoreauEnv(
    prob, rho=2*prob.L, epsilon=epsilon, 
    K=K, optim_algo=optim_algo, y_norm=y_norm,
    stepsize=1./2./max(prob.G, 3*prob.L)*(prob.m**0.5),
    epsilon_check_freq=epsilon_check_freq,
    log_freq=log_freq, log_prefix='moreau: ')
  return moreau_env


def create_init_log_dir(parent_log_dir, _log_dir):
  date_string = datetime.now().strftime("%Y%m%d-%H%M%S")
  _log_dir = '{}_{}'.format(_log_dir, date_string)
  log_dir = os.path.join(parent_log_dir, _log_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  utils.dump_script_folder(
    log_dir, temp_script_dir, date_string=date_string)
  return log_dir


def plot_proxfdiag(
  prob, output, epsilon, x_0, log_dir, _log_dir, label):
  np.savez(
    os.path.join(log_dir, 'output.npz'),
    epsilon=epsilon, x_0=x_0,
    **output)
  
  xk_list = np.array(output['xk_list'])
  nof_grads_list = np.array(output['nof_grads_list'])
  xk_inner_list = np.array(output['xk_inner_list'])
  moreau_grad_norm_list = np.array(output['moreau_grad_norm_list'])

  print('plotting started')
  plt.loglog(nof_grads_list, moreau_grad_norm_list, 'b-', label=label)
  plt.title('moreau grad')
  plt.legend()
  plt.savefig(os.path.join(
    log_dir, 'moreau_{}.png'.format(_log_dir)))
  plt.savefig(os.path.join(
    log_dir, 'moreau_{}.pdf'.format(_log_dir)), bbox_inches='tight')
  plt.close()

  func_value_list = []
  for x_k in xk_list:
    func_value_list.append(prob.func(x_k))
  func_value_inner_list = []
  for x_k in xk_inner_list:
    func_value_inner_list.append(prob.func(x_k))

  plt.loglog(nof_grads_list, func_value_list, 'b-', label='outer')
  plt.loglog(func_value_inner_list, 'r--', label='inner')
  plt.title('function value')
  plt.legend()
  plt.savefig(os.path.join(
    log_dir, 'function_value_{}.png'.format(_log_dir)))
  plt.savefig(os.path.join(
    log_dir, 'function_value_{}.pdf'.format(_log_dir)), bbox_inches='tight')
  plt.close()

  if d == 2:
    mode = 'contourf'
    utils.plot2d(
      prob.func, x_min=-5, x_max=5, x_num=100, 
      y_min=-5, y_max=5, y_num=100, 
      save_path=os.path.join(log_dir, 'func_2dplot_path'),
      mode=mode, line_curve=xk_list)
    plt.close()
  print('plotting done')    


def proxfdiag_expt(prob, epsilon, x_0):
  optim_algo = 'ProxFDIAG'
  parent_log_dir = os.path.join(_parent_log_dir, 'prox_fdiag')
  print('prox_fdiag, eps={}'.format(epsilon))

  moreau_env = generate_moreau_envelopes(prob)

  _log_dir = 'seed{}_d{}_m{}_{}_eps{:.3g}_x0_{}_{}'.format(
    seed, d, m, optim_algo, epsilon, x_0[0], x_0[1])
  log_dir = create_init_log_dir(parent_log_dir, _log_dir)

  output = algos.proxFDIAG_optimize(
    prob, epsilon, x_0=x_0, log_freq=1,
    excess_gap_log_freq=10000,
    moreau_env=moreau_env, moreau_stop=None)
  plot_proxfdiag(
    prob, output, epsilon, x_0, log_dir, _log_dir, label=optim_algo)


def adaptive_proxfdiag_expt(prob, epsilon, x_0):
  optim_algo = 'ProxFDIAG'
  parent_log_dir = os.path.join(_parent_log_dir, 'adaptive_prox_fdiag')
  print('prox_fdiag, eps={}'.format(epsilon))

  moreau_env = generate_moreau_envelopes(prob)

  epsilon_start=10.
  epsilon_division=2.0

  _log_dir = 'seed{}_d{}_m{}_{}_eps{:.3g}start{}div{}_x0_{}_{}'.format(
    seed, d, m, optim_algo, epsilon,
    epsilon_start, epsilon_division, x_0[0], x_0[1])
  log_dir = create_init_log_dir(parent_log_dir, _log_dir)

  output = algos.adaptiveProxFDIAG_optimize(
    prob, epsilon,
    epsilon_start=epsilon_start, epsilon_division=epsilon_division,
    x_0=x_0, log_freq=1,
    excess_gap_log_freq=10000,
    moreau_env=moreau_env, moreau_stop=True)

  plot_proxfdiag(
    prob, output, epsilon, x_0, log_dir, _log_dir, label=optim_algo)


def plot_subgrad(
  prob, output, K, stepsize_rule, stepsize,
  x_0, moreau_env, log_dir, _log_dir, label):
  sampled_moreau_grad_norm_list = []
  sampled_func_list = []
  best_func_list = []
  for kk in range(0, int(np.log10(K))):
    k = 10**kk
    stepsizes = np.power(np.arange(1, k+1), -0.5)
    sampling_prob = stepsizes/stepsizes.sum()
    sample_k = np.random.choice(k, p=sampling_prob)
    sampled_moreau_grad_norm_list.append(
      utils.l2_norm(
        moreau_env.grad(output['xk_list'][sample_k]))
      )
    sampled_func_list.append(
      prob.func(output['xk_list'][sample_k]))
    best_func_list.append(
      prob.func(output['xk_best_list'][1+(10**kk)]))
  output['sampled_moreau_grad_norm_list'] = sampled_moreau_grad_norm_list
  output['sampled_func_list'] = sampled_func_list
  output['best_func_list'] = best_func_list
  nof_grads_moreau_list = 10**np.arange(1, np.log10(K)+1)
  output['nof_grads_moreau_list'] = nof_grads_moreau_list

  np.savez(
    os.path.join(log_dir, 'output.npz'),
    K=K, x_0=x_0, 
    stepsize_rule=stepsize_rule, stepsize=stepsize,
    **output)

  print('plotting started')
  plt.loglog(nof_grads_moreau_list, sampled_moreau_grad_norm_list, 'b-', label=label)
  plt.title('sampled moreau grad')
  plt.legend()
  plt.savefig(os.path.join(
    log_dir, 'moreau_sampled_{}.png'.format(_log_dir)))
  plt.savefig(os.path.join(
    log_dir, 'moreau_sampled_{}.pdf'.format(_log_dir)), bbox_inches='tight')
  plt.close()

  plt.semilogx(nof_grads_moreau_list, sampled_func_list, 'b-', label=label)
  plt.title('sampled func value')
  plt.legend()
  plt.savefig(os.path.join(
    log_dir, 'func_sampled_{}.png'.format(_log_dir)))
  plt.savefig(os.path.join(
    log_dir, 'func_sampled_{}.pdf'.format(_log_dir)), bbox_inches='tight')
  plt.close()

  plt.semilogx(nof_grads_moreau_list, best_func_list, 'b-', label=label)
  plt.title('best func value')
  plt.legend()
  plt.savefig(os.path.join(
    log_dir, 'func_best_{}.png'.format(_log_dir)))
  plt.savefig(os.path.join(
    log_dir, 'func_best_{}.pdf'.format(_log_dir)), bbox_inches='tight')
  plt.close()

  if 'moreau_grad_norm_best_list' in output:
    moreau_grad_norm_best_list = output['moreau_grad_norm_best_list']
    if not len(moreau_grad_norm_best_list) == 0:
      plt.loglog(moreau_grad_norm_best_list, 'b-', label=label)
      plt.title('moreau grad')
      plt.legend()
      plt.savefig(os.path.join(
        log_dir, 'moreau_best_{}.png'.format(_log_dir)))
      plt.savefig(os.path.join(
        log_dir, 'moreau_best_{}.pdf'.format(_log_dir)), bbox_inches='tight')
      plt.close()

      best_moreau_grad_norm_best_list = [
        moreau_grad_norm_best_list[0]]
      for mgn in moreau_grad_norm_best_list[1:]:
        if mgn < best_moreau_grad_norm_best_list[-1]:
          best_moreau_grad_norm_best_list.append(mgn)
        else:
          best_moreau_grad_norm_best_list.append(
            best_moreau_grad_norm_best_list[-1])
      plt.loglog(best_moreau_grad_norm_best_list, 'b-', label=label)
      plt.title('moreau grad')
      plt.legend()
      plt.savefig(os.path.join(
        log_dir, 'best_moreau_best_{}.png'.format(_log_dir)))
      plt.savefig(os.path.join(
        log_dir, 'best_moreau_best_{}.pdf'.format(_log_dir)), bbox_inches='tight')
      plt.close()

  xk_list = np.array(output['xk_list'])
  if d == 2:
    mode = 'contourf'
    utils.plot2d(
      prob.func, x_min=-5, x_max=5, x_num=100, 
      y_min=-5, y_max=5, y_num=100, 
      save_path=os.path.join(log_dir, 'func_2dplot_path'),
      mode=mode, line_curve=xk_list)
    plt.close()
  print('plotting done')


def subgrad_expt(prob, K, x_0, dynamic_moreau_comp=False):
  optim_algo = 'SubGrad{}'.format(
    '_neurips' if dynamic_moreau_comp else '')
  parent_log_dir = os.path.join(_parent_log_dir, 'subgrad{}'.format(
    '_neurips' if dynamic_moreau_comp else ''))
  stepsize_rule = 'davis_gamma'

  stepsize = 0.1*prob.G*((prob.L**3)**0.5) # hardcoded stepsize for this problem

  moreau_env = generate_moreau_envelopes(prob)

  _log_dir = 'seed{}_d{}_m{}_{}_gamma{}_K{}_x0_{}_{}'.format(
    seed, d, m, optim_algo, stepsize, K, x_0[0], x_0[1])
  log_dir = create_init_log_dir(parent_log_dir, _log_dir)

  _moreau_env = moreau_env if dynamic_moreau_comp else None
  output = algos.subGradientDescent_optimize(
    prob, K=K, x_0=x_0, projection_fn=None,
    stepsize_rule=stepsize_rule, stepsize=stepsize,
    best_criteria='function', log_freq=10000,
    moreau_env=_moreau_env)
  
  plot_subgrad(
    prob, output, K, x_0, stepsize_rule, stepsize,
    moreau_env, log_dir, _log_dir, label=optim_algo)


def subgradvar_expt(prob, K, x_0, dynamic_moreau_comp=False):
  optim_algo = 'SubGradVar{}'.format(
    '_neurips' if dynamic_moreau_comp else '')
  parent_log_dir = os.path.join(_parent_log_dir, 'subgrad_var{}'.format(
    '_neurips' if dynamic_moreau_comp else ''))
  stepsize_rule = 'davis_gamma_var'

  stepsize = 0.1*prob.G*((prob.L**3)**0.5) # hardcoded stepsize for this problem
  moreau_env = generate_moreau_envelopes(prob)

  _log_dir = 'seed{}_d{}_m{}_{}_gamma{}_K{}_x0_{}_{}'.format(
    seed, d, m, optim_algo, stepsize, K, x_0[0], x_0[1])
  log_dir = create_init_log_dir(parent_log_dir, _log_dir)

  _moreau_env = moreau_env if dynamic_moreau_comp else None
  output = algos.subGradientDescent_optimize(
    prob, K=K, x_0=x_0, projection_fn=None,
    stepsize_rule=stepsize_rule, stepsize=stepsize,
    best_criteria='function', log_freq=10000,
    moreau_env=_moreau_env)
  
  plot_subgrad(
    prob, output, K, x_0, stepsize_rule, stepsize,
    moreau_env, log_dir, _log_dir, label=optim_algo)


if __name__ == '__main__':
  importlib.reload(algos)
  importlib.reload(probs)
  importlib.reload(utils)
  importlib.reload(morenv)

  seeds = range(1, 11)
  if len(sys.argv) == 1:
    optim_algos = [
      # 'ProxFDIAG', #algos.proxFDIAG_optimize
      'AdaptiveProxFDIAG', # algos.adaptiveProxFDIAG_optimize
      # 'SubGrad', # algos.subGradientDescent_optimize
      # 'SubGradVar', # algos.subGradientDescent_optimize
      # 'SubGrad_neurips', # algos.subGradientDescent_optimize
      # 'SubGradVar_neurips', # algos.subGradientDescent_optimize
    ]
  elif len(sys.argv) == 2:
    optim_algos = [sys.argv[1]]
  elif len(sys.argv) == 3:
    optim_algos = [sys.argv[1]]
    seeds = [int(sys.argv[2])]
  else:
    raise ValueError(
    'Please use the syntax: '
    'python expt_quadratic.py (ProxFDIAG|AdaptiveProxFDIAG'
    '|SubGrad|SubGradVar|SubGrad_neurips|SubGradVar_neurips) [seed]')


  d = 2
  x_0 = 4.*np.ones([d])
  m = 8
  for optim_algo in optim_algos:
    for seed in seeds:
      prob = generate_max_quadratic_prob(
        d=d, m=m, x_0=x_0, seed=seed)

      if optim_algo == 'ProxFDIAG':
        for epsilon in 10.**np.arange(0, -4, -1):
          proxfdiag_expt(prob, epsilon, x_0)
      elif optim_algo == 'AdaptiveProxFDIAG':
        epsilon = 10**-7
        adaptive_proxfdiag_expt(prob, epsilon, x_0)
      elif optim_algo == 'SubGrad':
        K = 10**7
        subgrad_expt(prob, K, x_0)
      elif optim_algo == 'SubGradVar':
        K = 10**7
        subgradvar_expt(prob, K, x_0)
      elif optim_algo == 'SubGrad_neurips':
        K = 10**7
        subgrad_expt(prob, K, x_0, dynamic_moreau_comp=True)
      elif optim_algo == 'SubGradVar_neurips':
        K = 10**7
        subgradvar_expt(prob, K, x_0, dynamic_moreau_comp=True)

  utils.remove_folder(temp_script_dir)