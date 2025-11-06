import numpy as np
from src.parameters import *
from src.CAModel import CAModel
from src.load_target import load_target
from src.loss import pixelWiseMSE
import matplotlib.pylab as plt

def plot_min_diff_f(loss_log, model, initial_step):
    x = np.arange(initial_step, initial_step+len(loss_log))
    plt.figure(figsize=(10, 4))
    plt.title(f"Minima diferencia a un fotograma (inicio={initial_step})")
    plt.plot(x, loss_log, '.', alpha=0.1)
    plt.savefig(f"{model}_min_diff_f.jpg")
    plt.close()

def plot_diff_f0(loss_log, model, initial_step):
    x = np.arange(initial_step, initial_step+len(loss_log))
    plt.figure(figsize=(10, 4))
    plt.title(f"Diferencia al primer fotograma (inicio={initial_step})")
    plt.plot(x, loss_log, '.', alpha=0.1)
    plt.savefig(f"{model}_diff_f0.jpg")
    plt.close()

def plot_rgb_space(RED_log, GREEN_log, BLUE_log, model, initial_step):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(RED_log, 
               GREEN_log, 
               BLUE_log,
               marker=('.'), s=np.repeat(2, len(RED_log)))
    #ax.plot3D(RED_log, GREEN_log, BLUE_log)
    ax.view_init(30, -120)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.title(f"RGB Space {model} (inicio={initial_step})")
    #plt.show()
    plt.savefig(f"{model}_RgbSpace.jpg")
    plt.close()

def get_ajust_step(ca, seed, periodo, f0):
    iter_n = 2*periodo # Le damos el tiempo de dos periodos para llegar al atractor

    x = np.repeat(seed[None, ...], 1, 0)
    error_anterior = np.array([])
    for i in range(iter_n):
        x = ca(x)
        dif_f0 = pixelWiseMSE(x[0], f0)
        error_anterior = np.append(error_anterior, dif_f0)
    
    # Como son dos periodos, deberia a hacercarse a cada fotograma por 2*T*TAU iteraciones
    min_indexes = np.argpartition(error_anterior,2*T*TAU)[:2*T*TAU]
    paso_ajuste = np.min(min_indexes)
    return paso_ajuste


def get_metrics(model_path, pad_target, iter_n):
    model = model_path.split("/")[-3]
    n_frames, h, w, c = pad_target.shape
    f0 = pad_target[0]
    
    PERIOD = n_frames*T*TAU

    ca = CAModel()
    ca.load_weights(model_path)

    seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
    seed[h//2, w//2, 3:] = 1.0

    x = np.repeat(seed[None, ...], 1, 0)

    # Logs
    error_log = np.array([])

    RED_log = np.array([])
    BLUE_log = np.array([])
    GREEN_log = np.array([])
    
    min_diff_f_log = np.array([])

    # Ejecutamos iteraciones hasta llegar al atractor
    paso_ajuste = get_ajust_step(ca, seed, PERIOD, f0)
    for i in range(paso_ajuste):
        x = ca(x)
    
    
    # Realizamos mediciones
    for i in range(iter_n):
        x = ca(x)
        
        error = pixelWiseMSE(x[0], f0)
        error_log = np.append(error_log, error.numpy())
        
        
        red = np.sum(x[0,:,:,0])
        blue = np.sum(x[0,:,:,1])
        green = np.sum(x[0,:,:,2])
        
        RED_log = np.append(RED_log, red)
        GREEN_log = np.append(GREEN_log, green)
        BLUE_log = np.append(BLUE_log, blue)
        
        min_diff_f = np.min([pixelWiseMSE(x[0], pad_target[t]) for t in range(len(pad_target))])
        min_diff_f_log = np.append(min_diff_f_log, min_diff_f)
        
        
        print("\r step: %d , Porcentaje completado: %04f"%(i, (i/iter_n)*100), end="")
        
    
    plot_diff_f0(error_log, model, paso_ajuste)
    plot_rgb_space(RED_log, GREEN_log, BLUE_log, model, paso_ajuste)
    plot_min_diff_f(min_diff_f_log, model, paso_ajuste)


    with open(f"{model}_metrics.txt", "w") as f:
        f.write(f"Paso de ajuste: {paso_ajuste}\n")
        f.write(f"Minima diferencia promedio a un fotograma: {np.mean(min_diff_f_log)}\n")
        

if __name__ == "__main__":
    model_path="models/3frames/3f_rgb_min/10000/10000.weights.h5"
    pad_target = load_target("data/Videos/heavy_diff_n=4.mp4")
    iter_n = 1_000
    get_metrics(model_path, pad_target, iter_n)