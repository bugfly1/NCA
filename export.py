from src.Utils import export_ca_to_webgl_demo
from src.CAModel import CAModel

path = "models/2000_rolling_loss/2000.weights.h5"

ca = CAModel()
ca.load_weights(path)
  
with open("ex_user.json", "w") as f:
  f.write(export_ca_to_webgl_demo(ca))
