from jax import random, vmap, jit
import jax.numpy as np
from jax.lax import log
from jax import linear_util as lu
from jax.tree_util import tree_map, tree_transpose, tree_structure
from jax._src.util import partial
from jax.api_util import argnums_partial
from jax.api import _unravel_array_into_pytree, _std_basis
from jax._src.api import _check_input_dtype_jacrev, _vjp, _check_output_dtype_jacrev
import numpy as old_np
from functools import lru_cache

"""
Calculate Sparse n-step pattern - Influence of parameters on hidden states in n timesteps.
"""
def calculateSnApPattern(snapLevel, weightRows, weightCols, recurrentRows, recurrentCols):

        @lru_cache(maxsize=None)
        def getInfluence(state):
            influence = np.where(recurrentRows == state) # where hidden state SnAP_rows[idx] influences other state
            influencedState = recurrentCols[influence] # next influenced state
            return influencedState

        SnAP_rows, SnAP_cols = [], []

        rows = np.concatenate((weightRows, recurrentRows))
        cols = np.concatenate((weightCols, recurrentCols))

        SnAP_rows.extend(cols[np.arange(len(rows))])
        SnAP_cols.extend(np.arange(len(rows)))

        if (snapLevel == 1):
            return SnAP_rows, SnAP_cols

        #reduce duplicates in recurrents
        coords = np.vstack((np.array(recurrentRows), np.array(recurrentCols)))
        [recurrentRows, recurrentCols] = old_np.unique(coords, axis=1)

        for s in range(1, snapLevel): #SnAP Level
            for idx in range(len(SnAP_rows)):
                influencedState = getInfluence(SnAP_rows[idx])
                SnAP_rows.extend(influencedState)
                SnAP_cols.extend(np.full((len(influencedState),), SnAP_cols[idx]))       

            coords = np.vstack((np.array(SnAP_rows), np.array(SnAP_cols)))
            [SnAP_rows, SnAP_cols] = old_np.unique(coords, axis=1)

            SnAP_rows = SnAP_rows.tolist()
            SnAP_cols = SnAP_cols.tolist()

        return np.array(SnAP_rows), np.array(SnAP_cols)

def jacrev(fun, argnums, holomorphic = False, allow_int = False):

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    jac = tree_transpose(tree_structure(example_args), tree_structure(y), jac)
    return jac, y

  return jacfun

class SparseMatrix:

    def __init__(self, key=random.PRNGKey(1), m=10, n=10, density=1, start=0):
        self.key = key
        self.density = density
        self.shape = (m, n) 
        self.start = start

    def jacobian(self, rows, cols, shape, start):
        self.rows = rows 
        self.cols = cols 
        self.shape = shape 
        self.start = start
        self.end = start + len(rows) 
        self.coords = (rows, cols)
        self.len = len(rows)
        self.density = self.len / (shape[0] * shape[1])
        
    def init(self):
        k1, k2 = random.split(self.key, 2)
        (m, n) = self.shape
        mn = m * n

        bound = np.sqrt(1/m)

        # Number of non zero values
        k = int(round(self.density * m * n))

        # flat index
        ind = random.choice(k1, mn, shape=(k,), replace=False).sort()

        row = np.floor(ind * 1. / n).astype(np.int16)
        col = (ind - row * n).astype(np.int16)
        #data = random.normal(self.key, (k,))
        data = random.uniform(self.key, (k,), minval=-bound, maxval=bound)

        self.rows = np.asarray(row) 
        self.cols = np.asarray(col)
        self.len = len(self.rows)
        self.end = self.start + self.len
        self.coords = (self.rows, self.cols)

        return np.asarray(data)


    @partial(jit, static_argnums=(0,))
    def toDense(self, data):
        return np.zeros(self.shape).at[tuple(self.coords)].add(data)

@jit
def BinaryCrossEntropyLoss(y_hat, y):
    loss =  -(y * log(y_hat) + (1-y)* log(1-y_hat))
    return np.mean(loss)

