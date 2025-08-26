#@title Train Utilities (SamplePool, Model Export, Damage)
import numpy as np

from src.parameters import *

class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def __str__(self):
    string = f"""Sample pool object: 
      IDs of sampled CA: {self._parent_idx},
      Amount of sampled CA: {self._size}
      """
    string += "slots:\n"
    for k in self._slot_names:
      value = getattr(self, k)
      if type(value) == np.ndarray:
        string += f"\t {k}: (np.array), {value.shape}"
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

    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  # Update samples from pool
  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)