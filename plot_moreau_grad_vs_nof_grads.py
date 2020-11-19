from datetime import datetime
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from scipy.stats import linregress

def find_dir_with_prefix(prefix, log_dir):
  for file in os.listdir(log_dir):
    if prefix in file:
      return os.path.join(log_dir, file)
  raise OSError('dir with prefix \'{}\' not found at: {}'.format(
    prefix, log_dir))

if __name__ == '__main__':
  if True:
    _parent_log_dir = 'results'
    subgradvar_log_dir = 'subgrad_var'
    subgradvar_prefix = 'd2_m8_SubGradVar_gamma1.1313708498984762_K10000000_x0_4.0_4.0'
    subgradvar_seeds = list(range(1, 11))

    subgradvar_moreau = []

    for seed in subgradvar_seeds:
      prefix = 'seed{}_{}'.format(seed, subgradvar_prefix)
      parent_log_dir = os.path.join(_parent_log_dir, subgradvar_log_dir)
      log_dir = find_dir_with_prefix(prefix, parent_log_dir)
      
      print('seed', seed, ':', log_dir)  

      npz_file = os.path.join(log_dir, 'output.npz')
      npz_file = np.load(npz_file)

      sampled_moreau_grad_norm_list = npz_file['sampled_moreau_grad_norm_list']
      nof_grads_moreau_list = npz_file['nof_grads_moreau_list']  
      subgradvar_moreau.append(sampled_moreau_grad_norm_list)

    subgradvar_moreau = np.array(subgradvar_moreau)

    plt.loglog(
      nof_grads_moreau_list, subgradvar_moreau.mean(axis=0), 
      'b^--', label='Sub-gradient method')
    y_error = stats.sem(subgradvar_moreau, axis=0)

    plt.errorbar(
      (nof_grads_moreau_list),
      (subgradvar_moreau).mean(axis=0), 
      yerr=y_error,
      # yerr=np.power(10., y_error),
      capsize=3,
      fmt = 'b^',
      )

    plt.legend()
    savepath = 'subgradvar_mean.png'

  if True:
    _parent_log_dir = 'results'
    proxfdiag_log_dir = 'prox_fdiag'
    proxfdiag_prefix = 'd2_m8_ProxFDIAG_eps'
    proxfdiag_seeds = list(range(1, 11))
    proxfdiag_epsilons = 10.**np.arange(0, -4, -1)

    proxfdiag_moreau_grad_norm = []
    proxfdiag_nof_inner_steps = []

    for epsilon in proxfdiag_epsilons:
      for seed in proxfdiag_seeds:
        prefix = 'seed{}_{}{:.3g}'.format(
          seed, proxfdiag_prefix, epsilon)
        parent_log_dir = os.path.join(_parent_log_dir, proxfdiag_log_dir)
        log_dir = find_dir_with_prefix(prefix, parent_log_dir)
        
        print('seed', seed, ':', log_dir)  

        npz_file = os.path.join(log_dir, 'output.npz')
        npz_file = np.load(npz_file)

        proxfdiag_moreau_grad_norm.append(npz_file['moreau_grad_norm_list'][-1])
        proxfdiag_nof_inner_steps.append(npz_file['nof_grads_list'][-1])

    plt.loglog(
      proxfdiag_nof_inner_steps, proxfdiag_moreau_grad_norm,
    'ro', label='Prox-FDIAG (ours)')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
      np.log10(proxfdiag_nof_inner_steps), np.log10(proxfdiag_moreau_grad_norm))

    inner_steps_range = np.arange(2, 8, 1)
    plt.plot(10.**inner_steps_range, 10.**(inner_steps_range*slope+intercept), 'r-')
    savepath = 'proxfdiag_mean'

  if True:
    _parent_log_dir = 'results'
    fastprox_log_dir = 'adaptive_prox_fdiag'
    fastprox_prefix = 'd2_m8_ProxFDIAG_eps'
    fastprox_seeds = list(range(1, 11))

    fastprox_moreau_grad_norm = []
    fastprox_nof_inner_steps = []

    for seed in fastprox_seeds:
      prefix = 'seed{}_{}'.format(seed, fastprox_prefix)
      parent_log_dir = os.path.join(_parent_log_dir, fastprox_log_dir)
      log_dir = find_dir_with_prefix(prefix, parent_log_dir)
      
      print('seed', seed, ':', log_dir)  

      npz_file = os.path.join(log_dir, 'output.npz')
      npz_file = np.load(npz_file)

      fastprox_moreau_grad_norm.extend(npz_file['moreau_grad_norm_list'][1:])
      fastprox_nof_inner_steps.extend(npz_file['nof_grads_list'][1:])

    fastprox_nof_inner_steps = np.array(fastprox_nof_inner_steps).reshape(-1,)
    fastprox_moreau_grad_norm = np.array(fastprox_moreau_grad_norm).reshape(-1,)

    subset = np.random.choice(
      len(fastprox_nof_inner_steps),
      int(len(fastprox_nof_inner_steps)))
    plt.loglog(
      fastprox_nof_inner_steps[subset], fastprox_moreau_grad_norm[subset],
      'k.', label='Adaptive Prox-FDIAG (ours)')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
      np.log10(fastprox_nof_inner_steps), np.log10(fastprox_moreau_grad_norm))

    inner_steps_range = np.arange(1, 7, 1)
    plt.plot(10.**inner_steps_range, 10.**(inner_steps_range*slope+intercept), 'k-')
    savepath = 'proxfdiag_mean'

  savepath = os.path.join('results', 'neurips_proxfdiag_vs_subgradvar')

  plt.legend()
  plt.savefig('{}.png'.format(savepath))
  plt.savefig('{}.pdf'.format(savepath), bbox_inches='tight')
  plt.close()