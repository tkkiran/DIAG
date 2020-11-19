import matplotlib
matplotlib.use('Agg')

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np

import glob, os, shutil, sys
from datetime import datetime

def l2_norm(x):
  return np.sqrt((np.array(x)**2).sum())


def inner_prod_sum(g_i, x_i):
  """
  sum over i of Inner product of g_i and x_i
  """
  return (g_i*x_i).sum(axis=1)


def exp_normalize(x):
  """
  Overflow safe exponential normalization
  https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
  """
  b = x.max()
  y = np.exp(x - b)
  return y / y.sum()


def generate_qudratic_fn(a, b, c=0.):
  """
  c + 0.5*a(x - b)**2
  """
  def f(x):
    return c+0.5*a*l2_norm(x-np.array(b))**2
  def g(x):
    return a*(x-np.array(b))
  return f, g


def projection_fn(x, x_origin, radius):
  x = x - x_origin
  if radius < l2_norm(x):
    x = x/l2_norm(x)*radius
  return x + x_origin


def project_to_simplex(vector, dimension=None):
  """
  Project real vectors {v_i} onto probability simplex as {p_i},
  such that sum_i p_i = 1

  Input
  x: 1-D array of size (dimension x number of vectors to project)
  
  Algorithm from:
  Efficient Projections onto the l 1 -Ball for Learning in High Dimensions,
  John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra 2008
  """
  if dimension is None:
    dimension = len(vector)

  # enforcing numpy array and reshaping to (num of vectors) x (dimension)
  # numpy stores array row-major wise
  x = np.array(vector).reshape([-1, dimension]).copy()

  sort_x = np.sort(x, axis=-1)[:, ::-1]  # sorted in descending order
  cumsum_x = sort_x.cumsum(axis=-1) - 1  # find cumulative sum shifted by -1
  # find first rho=1,2,3...d when sort_x[rho] < ((sum_{i=1}^rho x[i])-1)/rho
  rho = np.argmax(
    (sort_x - (cumsum_x/range(1,dimension+1))) < 0, axis=-1)

  # if rho = 0 => non negatives found, all dimension are satisfied => rho=d
  rho[rho == 0] = x.shape[-1]
  lambda_x = cumsum_x[range(x.shape[0]), rho-1]/rho

  return np.maximum(x.T - lambda_x, 0).T.ravel()

def mirror_descent_simplex(x, grad, stepsize):
  """
  KL divergence (Bregman divergence) based 
  Mirror-Descent update

  x_plus = sum_i e_i (x_i*exp(-stepsize*grad_i))/(sum_j x_j*exp(-stepsize*grad_j))
  """
  grad = np.array(grad)
  return exp_normalize(np.log(x) - stepsize*grad)

# def plot1(func, d, x_min, x_max, x_step=1.):
#   assert d == 1

#   xs = []
#   ys = []
#   _x = x_min
#   while _x <= x_max:
#     xs.append(_x)
#     ys.append(func(_x))
#     _x += x_step

#   plt.plot(xs, ys)
#   plt.show()
#   plt.close()

# def plot2(
#   func, d, x1_min, x1_max, x1_step=1., 
#   x2_min=None, x2_max=None, x2_step=None):
#   assert d == 2

#   if x2_min is None:
#     x2_min = x1_min
#   if x2_max is None:
#     x2_max = x1_max
#   if x2_step is None:
#     x2_step = x1_step

#   xs = []
#   ys = []
#   _x1 = x1_min
#   _x2 = x2_min
#   while _x1 <= x1_max:
#     ys.append([])
#     xs.append([])
#     _x2 = x2_min
#     while _x2 <= x2_max:
#       xs[-1].append((_x1, _x2))
#       ys[-1].append(func([_x1, _x2]))
#       _x2 += x2_step
#     _x1 += x1_step

#   plt.imshow(ys)
#   plt.show()
#   plt.close()



def plot2d(
  func, x_min, x_max, x_num, y_min, y_max, y_num, save_path, 
  mode='contourf', nbins=60, line_curve=None, show_fig=False):
  # make these larger to increase the resolution
  # x_num, y_num

  # generate 2 2d grids for the x & y bounds
  xx = np.linspace(x_min, x_max, x_num)
  yy = np.linspace(y_min, y_max, y_num)
  x, y = np.mgrid[x_min:x_max:x_num*1j, 
                  y_min:y_max:y_num*1j]

  z = np.zeros([len(xx), len(yy)])
  for i in range(len(xx)):
    for j in range(len(yy)):
      z[i, j] = func([xx[i], yy[j]])

  # x and y are bounds, so z should be the value *inside* those bounds.
  # Therefore, remove the last value from the z array.
  z = z[:-1, :-1]
  levels = MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())


  # pick the desired colormap, sensible levels, and define a normalization
  # instance which takes data values and translates those into levels.
  cmap = plt.get_cmap('PiYG')

  fig, ax1 = plt.subplots()

  if mode == 'pcolormesh':
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax1.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    # ax0.set_title('pcolormesh with levels')
  elif mode == 'contourf':
    # contours are *point* based plots, so convert our bound into point
    # centers
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                      y[:-1, :-1] + dy/2., z, levels=levels,
                      cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    if line_curve is not None:
      line_curve = np.array(line_curve)
      plt.plot(line_curve[:, 0], line_curve[:, 1], 'b--', alpha=0.25)
      plt.plot(line_curve[:, 0], line_curve[:, 1], 'b.')
    # ax1.set_title('contourf with levels')

  # adjust spacing between subplots so `ax1` title and `ax0` tick labels
  # don't overlap
  fig.tight_layout()
  plt.savefig('{}_{}.png'.format(save_path, mode))
  plt.savefig('{}_{}.pdf'.format(save_path, mode))
  if show_fig:
    plt.show()
  plt.close()


def dump_script(dirname, script_file=None, file_list=None):
  dest = os.path.join(dirname, 'script_{}'.format(
    datetime.now().strftime("%Y%m%d-%H%M%S")))
  os.mkdir(dest)
  print('copying files to {}'.format(dest))
  if file_list is None:
    file_list = glob.glob("*.py")
  for file in file_list:
    _dest = os.path.join(dest, os.path.dirname(file))
    print('copying {} to {}'.format(file, _dest))
    if not os.path.exists(_dest):
      os.makedirs(_dest)
    shutil.copy2(file, _dest)
  if script_file is not None:
    print('copying {}'.format(script_file))
    shutil.copy2(script_file, dest)

  with open(os.path.join(dest, "command.txt"), "w") as f:
      f.write(" ".join(sys.argv) + "\n")
  
  return dest

def dump_script_folder(dirname, source_dir, date_string=None):
  if date_string is None:
    date_string = datetime.now().strftime("%Y%m%d-%H%M%S")
  dest = os.path.join(dirname, 'script_{}'.format(date_string))
  print('copying script folder to {}'.format(dest))
  shutil.copytree(source_dir, dest)

def remove_folder(dirname):
  shutil.rmtree(dirname)