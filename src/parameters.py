# ==================== Parametros ================================

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 4    # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8     
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

# Extension de serie temporal
T = 1

# Numero de iteraciones por frame o "calibracion de relojes internos"
ITER_FRAME = 16

N_ITER_CA = [64, 96]     # [64, 96] por defecto

SRC_TARGET = "data/Videos/heavy_difference.mp4"
#SRC_TARGET = "data/images/charmander.png"

START_TRAINING_FROM_SAVE_POINT = False
SAVE_POINT = 7000 # step

EXPERIMENT_TYPE = "Serie" #@param ["Growing", "Persistent", "Regenerating", "Roll"]

# Experimentos:
# Growing, Persistent, Regenerating: Los mismos que se encuentran en el distill
# Roll: Primera implementacion de videos como inputs, cada batch son todos los 
#       frames del video y se comparan con el frame siguiente en loss_f()

EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2, "Roll": 3, "Serie": 4}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1, 1, 0][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3, 3, 0][EXPERIMENT_N]  # Number of patterns to damage in a batch
VIDEO = [0, 0, 0, 1, 1][EXPERIMENT_N]
ROLL =  [0, 0, 0, 1, 0][EXPERIMENT_N]
SERIE = [0, 0, 0, 0, 1][EXPERIMENT_N]

