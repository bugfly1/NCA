import numpy as np

from src.parameters import ROLL

class SamplePool:
  def __init__(self, x=None, _parent=None, _parent_idx=None):
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
    idx = np.random.choice(self._size, n, False)

    batch = self.x[idx]
    batch = SamplePool(x=batch, _parent=self, _parent_idx=idx)
    return batch

  # Update samples from pool
  def commit(self):
    self._parent.x[self._parent_idx] = self.x
    
## WIP (para ROLL)
class SequencePool:
    def __init__(self, n_frames, seed, pool_size, x=None, _parent=None, _idx_sons=None, _pools=None):
        self._parent = _parent
        self._pools = _pools
        self.x = x
        if _pools == None:
            self._pools = []
            for i in range(n_frames):
                pool = SamplePool(x = np.repeat(seed, pool_size, 0))
                self._pools.append(pool)
        
    def sample(self, n):
        H, W = 48, 48
        CHANNEL_N = 16
        n_frames = 2
        x = np.zeros((n_frames, H, W, CHANNEL_N))
        batch = []
        for i in range(len(self._sons)):
            batch.append(self._sons[i].sample(1))
            x[i] = batch.x
        idx_sons = [batch._parent_idx for pool in batch]
        
        SequencePool(x=x, _parent=self, _idx_sons=None)
        return batch
    
    def commit(self): 
        for i in range(len(self._parent._pools)):
            self._parent._pools[i].x = self.x[-1+i]
            self._parent._pools.commit()
        return    