#@title Train Utilities (SamplePool, Model Export, Damage)
import tensorflow as tf
import numpy as np

from src.parameters import *

## Esta demasiado generalizadaaaaaaa
# Todos los batch (resultados de .sample()) son hijos del mismo pool
# y este solamente tiene como slot 'x', los hijos tendran de slots
# su propio batch llamado x con 8 de los 1024 que tenemos


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
      _parent_idx: {self._parent_idx},
      _size: {self._size}
      """
    string += "slots:\n"
    for k in self._slot_names:
      value = getattr(self, k)
      if type(value) == np.ndarray:
        string += f"\t {k}: (np.array), {value.shape}"
    return string

  # Select n samples from pool
  def sample(self, n):
    # Select random samples
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  # Update samples from pool
  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)