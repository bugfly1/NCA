import tensorflow as tf
import numpy as np

# ==================== Parametros ================================

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 2    # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

# Extension de serie temporal
TAU = 3

# Numero de iteraciones por frame o "calibracion de relojes internos"
T = 16

# Beta, softmin
# b=87.0 no explota en el intervalo [0, 1] con precision tf.float32
# b=744.0 no explota en el intervalo [0, 1] con precision tf.float64
b = 200.0

# Precision de valores, float32 es mas rapido pero con float64 se pueden usar mayores valores de b,
# lo que de da prioridad a una sola sequencia gracias a softmin
PRECISION=tf.float32

# Delta, Huber Loss
delta = 0.75

# Determina si se utiliza el canal alpha para perdida y para vida de la celdas
ALPHA = False

SRC_TARGET = "data/Videos/translation.mp4"


if PRECISION == tf.float64:
    NP_PRECISION = np.float64
else:
    NP_PRECISION = np.float32

N_ITER_CA = [64, 96]     # [64, 96] por defecto

START_TRAINING_FROM_SAVE_POINT = False
SAVE_POINT = 15000 # step

EXPERIMENT_TYPE = "Serie" #@param ["Growing", "Persistent", "Regenerating", "Roll"]

# Experimentos:
# Growing, Persistent, Regenerating: Los mismos que se encuentran en el distill
# Roll: Primera implementacion de videos como inputs, cada batch son todos los 
#       frames del video y se comparan con el frame siguiente en loss_f()
# Serie: ejecutamos varias iteraciones de

EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2, "Roll": 3, "Serie": 4, "SerieCorta": 5}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1, 1, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3, 3, 0, 0][EXPERIMENT_N]  # Number of patterns to damage in a batch
VIDEO = [0, 0, 0, 1, 1, 1][EXPERIMENT_N]
ROLL =  [0, 0, 0, 1, 0, 0][EXPERIMENT_N]
SERIE = [0, 0, 0, 0, 1, 1][EXPERIMENT_N]
SERIE_CORTA = [0, 0, 0, 0, 0, 1][EXPERIMENT_N]
