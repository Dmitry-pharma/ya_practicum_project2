from utils2 import train,plot_final_comparison
import torch
from config import Config 


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
cfg = Config()

visualizer =train(cfg, device)

# Показать итоговые графики
plot_final_comparison(visualizer)