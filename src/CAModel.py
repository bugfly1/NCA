import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

from src.parameters import CHANNEL_N, CELL_FIRE_RATE, PRECISION, ALPHA

def get_living_mask(x):
  alpha = x[:, :, :, 3:4]
  # Cell is considered empty if there is no alpha > 0.1 cell in its
  # 3x3 neightborhood  
  return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1


class CAModel(tf.keras.Model):
  def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
    super().__init__(dtype=PRECISION)
    self.channel_n = channel_n
    self.fire_rate = fire_rate

    self.dmodel = tf.keras.Sequential([
            Conv2D(128, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                kernel_initializer=tf.zeros_initializer),
    ])

    self(tf.zeros([1, 3, 3, channel_n], dtype=PRECISION))  # dummy call to build the model
  
  @tf.function
  def perceive(self, x, angle=0.0):
    
    if PRECISION == tf.float64:
        identify = np.float64([ [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
        laplacian = np.float64([   [1, 2, 1],   # No entiendo porque se divide por 4 pero aqui lo hacen 
                                   [2,-12,2],   # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011589 p.16
                                   [1, 2, 1]])/ 4.0
        # Sobel_x
        dx = np.float64([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) / 8.0
    else:
        identify = np.float32([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        laplacian = np.float32([   [1, 2, 1],
                                   [2,-12,2],
                                   [1, 2, 1]]) / 4.0
        # Sobel_x
        dx = np.float32([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) / 8.0
    # Sobel_y  
    dy = dx.T
    c, s = tf.cos(angle), tf.sin(angle)
    c, s = tf.cast(c, PRECISION), tf.cast(s, PRECISION)
    
    # By convolution we stack each cell state, its partial derivatives in x and y
    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    
    kernel = tf.repeat(kernel, self.channel_n, 2)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    return y

  @tf.function
  def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
    if ALPHA:
        pre_life_mask = get_living_mask(x)

    # We stack the dx and dy values to each cell
    y = self.perceive(x, angle)
    
    # We run the CA convolution
    step_size = tf.cast(step_size, PRECISION)
    dx = self.dmodel(y)*step_size
    if fire_rate is None:
      fire_rate = self.fire_rate
      
    # Which cell are gonna get updated (Stochastic cell update)
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
    
    # We apply the change on updated cells
    x += dx * tf.cast(update_mask, PRECISION)

    if ALPHA:
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, PRECISION)
    
    return x


class CA1DModel(tf.keras.Model):
  def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE, R=2):
    super().__init__(dtype=PRECISION)
    self.channel_n = channel_n
    self.fire_rate = fire_rate
    self.R = R
    assert R > 0 and R < 100

    self.dmodel = tf.keras.Sequential([
            Conv2D(128, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None,
                kernel_initializer=tf.zeros_initializer),
    ])

    self(tf.zeros([1, 1, 3, channel_n], dtype=PRECISION))  # dummy call to build the model


  @tf.function
  def perceive(self, x, angle=0.0):    
    filter_lenght = 2*self.R + 1
    identify = np.zeros([1, filter_lenght], dtype=np.float32)
    dx = np.zeros([1, filter_lenght], dtype=np.float32)
    
    identify[0, filter_lenght//2] = 1
    
    #identify = np.float32([[0, 0, 1, 0, 0]])
    # Sobel_x
    for i in range(self.R):
        dx[0, i] = -(i +1)
        dx[0, (filter_lenght - 1) - i] = i + 1
    dx = dx/np.sum(np.abs(dx))
        
    #dx = np.float32([[-1, -2, 0, 2, 1]]) / 6.0
    # By convolution we stack each cell state, its partial derivatives in x and y
    kernel = tf.stack([identify, dx], -1)[:, :, None, :]
    
    kernel = tf.repeat(kernel, self.channel_n, 2)
    
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    return y

  @tf.function
  def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
    if ALPHA:
        pre_life_mask = get_living_mask(x)

    # We stack the dx and dy values to each cell
    y = self.perceive(x, angle)
    
    # We run the CA convolution
    step_size = tf.cast(step_size, PRECISION)
    dx = self.dmodel(y)*step_size
    if fire_rate is None:
      fire_rate = self.fire_rate
      
    # Which cell are gonna get updated (Stochastic cell update)
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
    
    # We apply the change on updated cells
    x += dx * tf.cast(update_mask, PRECISION)

    if ALPHA:
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, PRECISION)
    
    return x