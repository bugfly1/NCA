from src.Utils import export_ca_to_webgl_demo
import zipfile
from src.CAModel import CAModel

ca = CAModel()
ca.load_weights("models/2000_rolling_loss/2000.weights.h5")

#  ======================= Export =======================
with zipfile.ZipFile('webgl_models8.zip', 'w') as zf:
  zf.writestr("ex_user.json", export_ca_to_webgl_demo(ca))
