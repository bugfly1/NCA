from src.Utils import export_ca_to_webgl_demo
from src.CAModel import CAModel

path = "models/8000_control/8000.weights.h5"

ca = CAModel()
ca.load_weights(path)
  
with open("ex_user.json", "w") as f:
  f.write(export_ca_to_webgl_demo(ca))
