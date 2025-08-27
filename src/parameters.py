# ==================== Parametros ================================

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8     
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

N_ITER_CA = [64, 96]     # [64, 96] por defecto

SRC_IMAGE = "data/images/dcc_comprimido.png"

# Ahora mismo, recibe un directorio con imagenes que considera un video,
# en Utils hay funciones para usar mp4 y gifs pero esto ayuda a controlar 
# la cantidad de frames
SRC_VIDEO = "data/Video"
MAX_FRAMES = 10

START_TRAINING_FROM_SAVE_POINT = False
SAVE_POINT = 7000 # step

EXPERIMENT_TYPE = "Regenerating" #@param ["Growing", "Persistent", "Regenerating", "Roll"]

# Experimentos:
# Growing, Persistent, Regenerating: Los mismos que se encuentran en el distill
# Roll: Primera implementacion de videos como inputs, cada batch son todos los 
#       frames del video y se comparan con el frame siguiente en loss_f(), 
#       no parece dar resultados

EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2, "Roll": 3}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

VIDEO = [0, 0, 0, 1][EXPERIMENT_N]
ROLL =  [0, 0, 0, 1][EXPERIMENT_N]

