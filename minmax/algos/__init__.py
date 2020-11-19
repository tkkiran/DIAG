import importlib

from . proxfdiag import proxFDIAG_optimize
from . proxfdiag import adaptiveProxFDIAG_optimize

from . subgrad import subGradientDescent_optimize

from . mirror_prox import finiteMirrorProx_optimize
from . primal_dual import finiteAccelPrimalDual_optimize

from . import proxfdiag as _module
importlib.reload(_module)
proxFDIAG_optimize = _module.proxFDIAG_optimize
fastProxFDIAG_optimize = _module.adaptiveProxFDIAG_optimize
del _module

from . import subgrad as _module
importlib.reload(_module)
subGradientDescent_optimize = _module.subGradientDescent_optimize
del _module

from . import mirror_prox as _module
importlib.reload(_module)
finiteMirrorProx_optimize = _module.finiteMirrorProx_optimize
del _module

from . import primal_dual as _module
importlib.reload(_module)
finiteAccelPrimalDual_optimize = _module.finiteAccelPrimalDual_optimize
del _module