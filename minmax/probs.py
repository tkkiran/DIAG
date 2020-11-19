import importlib
import numpy as np

from . import utils

importlib.reload(utils)


# class NonsmoothConvexProb(object):
#   """
#   Nonsmooth Convex problem:
#   \\min_x f(x)
#   """
#   def __init__(self, d, func, subgrad, G=None, D=None):
#     self.d = d
#     self._func = func
#     self._subgrad = subgrad

#     self.G = None if G is None else float(G)
#     self.D = None if D is None else float(D)

#   def func(self, x):
#     return self._func(x)

#   def subgrad(self, x):
#     return self._subgrad(x)


class AffineQuadraticMinimaxProb(object):
  """
  Minimize x \\in \\reals^d:
  f(x) = 0.5 ||x - x_f||^2 + \\max_{1<=i<=m} [f_i + <g_i, x - x_i>]

  g(x, u) = 0.5 ||x - x_f||^2 + \\sum_{1<=i<=m} u_i [f_i + <g_i, x - x_i>]
  phi(u) = \\min_x 0.5 ||x - x_f||^2 + \\sum_{1<=i<=m} u_i [f_i + <g_i, x - x_i>]

  Attributes:
    m: number of functions
    d: dimension of the variable
    g_i: np.array([m, d])
    f_i: np.array([m])
    x_i: np.array([m, d])
    x_f: np.array([d])
    sigma: real number
  """
  def __init__(self, g_i, f_i=None, x_i=None, x_f=None, sigma=1.):
    """
    Args:
      g_i: np.array([m]) or np.array([m, d])
      f_i: None ([0, ..., 0]) or np.array([m]) or np.array([m, 1])
      x_i: None ([[0, ..., 0], ..., [0, ..., 0]]) or np.array([m, d]) or np.array([d])
      x_f: None ([0, ..., 0]) or np.array([d]) or np.array([1, d])
      sigma: None (1) or real number
    """
    self.g_i = np.array(g_i)
    self.m = self.g_i.shape[0]
    self.g_i = self.g_i.reshape(self.m, -1)
    self.d = self.g_i.shape[1] 

    if x_i is None:
      x_i = [0.]*self.d
    self.x_i = np.array(x_i).reshape(-1, self.d)
    if f_i is None:
      f_i = [0.]*self.m
    self.f_i = np.array(f_i).reshape(self.m,)

    self.sigma = float(sigma)
    if x_f is None:
      x_f = [0.]*self.d
    self.x_f = np.array(x_f).reshape(self.d,)

  def ip(self, g_i, x_i):
    """
    Computes inner product of g_i and x_i for different i
  
    Args:
      g_i: np.array([m, d])
      x_i: np.array([m, d])

    Returns:
      [(g_i[i]*x_i[i]).sum() for i in m]
      vector of size m
    """
    return (g_i*x_i).sum(axis=-1)

  def f(self, x):
    """
    Computes the function value
    f(x) = 0.5 ||x - x_f||^2 + \\max_{1<=i<=m} [f_i + <g_i, x - x_i>]
  
    Args:
      x: np.array([d])

    Returns:
      f(x): real function value
    """
    return (0.5*self.sigma*utils.l2_norm(x - self.x_f)**2 + 
            np.max(self.f_i + self.ip(self.g_i, x - self.x_i) ))

  def x0(self, u):
    """
    Minimizer of g(x, u) over x

    [N]: Eq. (2.9) on Pg. 238

    [N] Nesterov, Yu. \"Excessive gap technique in nonsmooth convex minimization.\"
        SIAM Journal on Optimization 16.1 (2005): 235-249.

    Args:
      u: np.array([m])

    Returns:
      x: np.array([d])
    """
    return self.x_f - self.g_i.T.dot(u)/self.sigma
  def x0_2(self, u):
    """
    Duplicate of self.x0(u)

    Args:
      u: np.array([m])

    Returns:
      x: np.array([d])
    """
    return self.x_f - self.ip(self.g_i.T, u)/self.sigma

  def phi(self, u):
    """
    phi(u) = min_x g(x, u)

    Args:
      u: np.array([m])

    Returns:
      phi(u): real number
    """
    return (0.5*self.sigma*utils.l2_norm(self.x0(u) - self.x_f)**2 + 
            self.ip(u, (self.f_i + self.ip(self.g_i, self.x0(u) - self.x_i) )))

  # def plot1(self, x_min, x_max, x_step=1.):
  #   assert self.d == 1

  #   xs = []
  #   ys = []
  #   _x = x_min
  #   while _x <= x_max:
  #     xs.append(_x)
  #     ys.append(self.f(_x))
  #     _x += x_step

  #   plt.plot(xs, ys)
  #   plt.show()
  #   plt.close()

  # def plot2(
  #   self, x1_min, x1_max, x1_step=1., 
  #   x2_min=None, x2_max=None, x2_step=None):
  #   assert self.d == 2

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
  #       ys[-1].append(self.f([_x1, _x2]))
  #       _x2 += x2_step
  #     _x1 += x1_step

  #   plt.imshow(ys)
  #   plt.show()
  #   plt.close()


class FiniteMinimaxProb(object):
  """
  Finite Minimax:
  \\min_x \\max_i f_i(x)

  f_i is L-smooth, G-Lipschitz continuous 
  sigma-strongly convex
  """
  def __init__(self, d, funcs, grads, L, G=None, D=None, sigma=None):
    self.d = d
    self._funcs = funcs
    self._grads = grads
    self.m = len(funcs)

    self.L = float(L)
    self.G = None if G is None else float(G)
    self.D = None if D is None else float(D)
    self.sigma = 0 if sigma is None else float(sigma)

  def func(self, x):
    return max([self.func_i(x, i) for i in range(self.m)])

  def func_i(self, x, i):
    return self._funcs[i](x)

  def funcs(self, x):
    return [self.func_i(x, i) for i in range(self.m)]

  def max_i(self, x):
    return np.argmax(self.funcs(x))

  def grad_i(self, x, i):
    return self._grads[i](x)

  def grads(self, x):
    return [self.grad_i(x, i) for i in range(self.m)]

  def subgrad(self, x):
    return self.grad_i(x, self.max_i(x))
