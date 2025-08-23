# ==================== Parametros ================================

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

EXPERIMENT_TYPE = "Video" #@param ["Growing", "Persistent", "Regenerating", new: "Video"]
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2, "Video":3}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

VIDEO = [0, 0, 0, 1][EXPERIMENT_N]

SRC_IMAGE = "data/images/dcc_comprimido.png"
SRC_VIDEO = "data/Video"

START_TRAINING_FROM_SAVE_POINT = False
SAVE_POINT = 7000 # step