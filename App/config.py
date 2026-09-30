"""Shared hyperparameters and constants."""
import torch

# ----------------- Environment -----------------
BLOCK_SIZE = 20
WINDOW_W = 640
WINDOW_H = 480

# ----------------- Session defaults (overridable from the CLI) -----------------
DEFAULT_STEPS = 30000   # 每局的總步數，約等於 100 FPS 下 5 分鐘
DEFAULT_FPS = 100       # 顯示畫面時的更新速度
DEFAULT_SEED = 42

# ----------------- DQN -----------------
STATE_DIM = 15
N_ACTIONS = 3           # straight, right turn, left turn
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 20000
TARGET_UPDATE_FREQ = 1000
TRAIN_START = 500
TRAIN_EVERY = 4

# ----------------- Exploration -----------------
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.8   # 從頭訓練時，epsilon 在前 80% 的步數內線性下降
EPS_FINETUNE = 0.05        # 接續訓練時固定的 epsilon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
