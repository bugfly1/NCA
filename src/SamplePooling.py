import numpy as np

from src.parameters import ROLL

class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, x=None):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._size = len(x)
    self.x = x

  def __str__(self):
    string = f"""Sample pool object: 
      IDs of sampled CA: {self._parent_idx},
      Amount of sampled CA: {self._size}
      """
    string += f"x = {self.x.shape}"
    return string

  # Select n samples from pool
  def sample(self, n):
    if ROLL:
      idx = np.arange(n, dtype=np.int32)
      idx *= np.int32(np.full((n), self._size / n))
      offset = np.random.choice(int(self._size / n))
      idx +=  np.repeat(offset, n)
      idx = idx.astype(np.int32)
    else:
      idx = np.random.choice(self._size, n, False)

    batch = self.x[idx]
    batch = SamplePool(x=batch, _parent=self, _parent_idx=idx)
    return batch

  # Update samples from pool
  def commit(self):
    self._parent.x[self._parent_idx] = self.x