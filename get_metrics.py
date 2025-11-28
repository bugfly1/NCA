import numpy as np
from src.parameters import *
from src.CAModel import CAModel
from src.load_target import load_target
from src.loss import pixelWiseMSE
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

from src.Utils import imwrite, to_rgb

mpl.rcParams['figure.dpi'] = 600

def plot_multiple_3D_pca(model, pcas):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for pca in pcas:
        ax.plot3D(
            pca[:, 0],
            pca[:, 1],
            pca[:, 2]
        )

        x, y, z = pca[0, :]
        ax.scatter3D(x, y, z, color='red', marker='o', s=100, label='Initial Point')
    
    ax.view_init(30, -120)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    k = len(pcas)
    plt.title(f"PCA projection {model}")
    plt.savefig(f"{model}_k={k}_PCA3D_projection.jpg")
    plt.close()
    

def plot_multiple_2D_pca(model, pcas):
    fig = plt.figure()
    ax = fig.add_subplot()
    
    for pca in pcas:
        ax.plot(
            pca[:, 0],
            pca[:, 1],
        )
    
    
        x, y = pca[0, :2]
        ax.scatter(x, y, color='red', marker='o', s=100, label='Punto inicial')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    k = len(pcas)
    plt.title(f"Proyeccion PCA")
    plt.savefig(f"{model}_k={k}_PCA2D_projection.jpg")
    plt.close()
    
    

def plot_2D_PCA(pca_result, model, paso_ajuste=0):
    fig = plt.figure()
    ax = fig.add_subplot()
    
    #ax.scatter(
    #    pca_result[:, 0],
    #    pca_result[:, 1],
    #    s=50, alpha=0.7
    #)
    ax.plot(
        pca_result[:, 0],
        pca_result[:, 1],
    )
    
    x, y = pca_result[0, :2]
    ax.scatter(x, y, color='red', marker='o', s=100, label='Punto inicial')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if PASO_AJUSTE:
        plt.title(f"PCA projection {model} (inicio={paso_ajuste})")
    else:
        plt.title(f"PCA projection {model}")
    plt.savefig(f"{model}_PCA2D_projection.jpg")
    plt.close()

def plot_3D_PCA(pca_result, model, paso_ajuste=0):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    #ax.scatter(
    #    pca_result[:, 0],
    #    pca_result[:, 1],
    #    pca_result[:, 2],
    #    s=50, alpha=0.7
    #)
    ax.plot3D(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
    )
    x, y, z = pca_result[0, :]
    ax.scatter3D(x, y, z, color='red', marker='o', s=100, label='Initial Point')
    
    ax.view_init(30, -120)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    if PASO_AJUSTE:
        plt.title(f"PCA projection {model} (inicio={paso_ajuste})")
    else:
        plt.title(f"PCA projection {model}")
    plt.savefig(f"{model}_PCA3D_projection.jpg")
    plt.close()

def pca_from_matrix(PCA_matrix, dims=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(PCA_matrix)
    pca = PCA(n_components=dims)
    pca_result = pca.fit_transform(scaled_data)
    return pca_result, pca.explained_variance_ratio_


def get_ajust_step(ca, seed, periodo, f0):
    n_periodos = 2
    iter_n = n_periodos*periodo # Le damos el tiempo de dos periodos para llegar al atractor

    x = np.repeat(seed[None, ...], 1, 0)
    error_anterior = np.array([])
    for i in range(iter_n):
        x = ca(x)
        dif_f0 = pixelWiseMSE(x[0], f0)
        error_anterior = np.append(error_anterior, dif_f0)
    
    min_indexes = np.argpartition(error_anterior, periodo)[:periodo]
    paso_ajuste = np.min(min_indexes)
    return paso_ajuste


def get_metrics(model_path, pad_target, iter_n, PASO_AJUSTE):
    model = model_path.split("/")[-3]
    n_frames, h, w, c = pad_target.shape
    f0 = pad_target[0]
    
    PERIOD = n_frames*T*TAU

    ca = CAModel()
    ca.load_weights(model_path)

    seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
    seed[h//2, w//2, 3:] = 1.0

    x = np.repeat(seed[None, ...], 1, 0)

    PCA_matrix = np.zeros((iter_n, h*w*CHANNEL_N))

    # Ejecutamos iteraciones hasta llegar al atractor
    if PASO_AJUSTE:
        paso_ajuste = get_ajust_step(ca, seed, PERIOD, f0)
        for i in range(paso_ajuste):
            x = ca(x)
    else: 
        paso_ajuste=0
    
    # Realizamos mediciones
    for i in range(iter_n):
        x = ca(x)
        flattened_vector = x.numpy().flatten()
        PCA_matrix[i] = flattened_vector
        
        print("\r step: %d , Porcentaje completado: %.3f"%(i, (i/iter_n)*100), end="")

    pca, variacion_por_eje = pca_from_matrix(PCA_matrix)
    plot_3D_PCA(pca, model, paso_ajuste)
    plot_2D_PCA(pca, model, paso_ajuste)
    
    with open(f"{model}_metrics.txt", "w") as f:
        f.write(f"Paso de ajuste: {paso_ajuste}\n")
        f.write(f"PCA Variacion por eje 3D: {variacion_por_eje}\n")


def get_sequence(model_path: str, pad_target, iter_n, PASO_AJUSTE):
    model = model_path.split("/")[-3]
    n_frames, h, w, c = pad_target.shape
    f0 = pad_target[0]
    
    PERIOD = n_frames*T*TAU

    ca = CAModel()
    ca.load_weights(model_path)

    #seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
    #seed[h//2, w//2, 3:] = 1.0
    
    seed = np.random.rand(h, w, CHANNEL_N)
    
    x = np.repeat(seed[None, ...], 1, 0)
    x = tf.cast(x, dtype=tf.float32)
    
    # Ejecutamos iteraciones hasta llegar al atractor
    if PASO_AJUSTE:
        paso_ajuste = get_ajust_step(ca, seed, PERIOD, f0)
        for i in range(paso_ajuste):
            x = ca(x)
    else: 
        paso_ajuste=0
    
    sequence = np.zeros([iter_n, h, w, CHANNEL_N], dtype=np.float32)
    # Realizamos mediciones
    for i in range(iter_n):
        sequence[i] = x.numpy()
        x = ca(x)
        print("\r step: %d , Porcentaje completado: %.3f"%(i, (i/iter_n)*100), end="")
        
    return sequence, paso_ajuste

def sequence_PCA(sequence, dims=3, paso_ajuste=0):
    n, h, w, c = sequence.shape
    PCA_matrix = np.zeros([n, h*w*c], dtype=np.float32)
    
    for i in range(n):
        flattened_vector = sequence[i].flatten()
        PCA_matrix[i] = flattened_vector
    
    pca, _ = pca_from_matrix(PCA_matrix)
    return pca

def period_states_image(model_path: str, pad_target, iter_n, PASO_AJUSTE):
    model = model_path.split("/")[-3]
    n_frames, h, w, c = pad_target.shape
    f0 = pad_target[0]
    
    PERIOD = n_frames*T*TAU

    ca = CAModel()
    ca.load_weights(model_path)

    seed = np.zeros([h, w, CHANNEL_N], dtype=np.float32)
    seed[h//2, w//2, 3:] = 1.0
    
    x = np.repeat(seed[None, ...], 1, 0)

    # Ejecutamos iteraciones hasta llegar al atractor
    if PASO_AJUSTE:
        paso_ajuste = get_ajust_step(ca, seed, PERIOD, f0)
        for i in range(paso_ajuste):
            x = ca(x)
    else: 
        paso_ajuste=0
    
    sequence = np.zeros([iter_n, h, w, 3], dtype=np.float32)
    # Realizamos mediciones
    for i in range(iter_n * PERIOD):
        if i % PERIOD == 0:
            sequence[i] = to_rgb(x).numpy()
        
        x = ca(x)
        print("\r step: %d , Porcentaje completado: %.3f"%(i, (i/iter_n)*100), end="")
    
    vis = np.hstack(sequence)
    imwrite('Secuencia_estados_periodo.jpg', vis)
    
    
    return sequence


def multiple_PCA_plots(model_path, k, pad_target, iter_n, PASO_AJUSTE):
    pcas = [0]*k
    for i in range(k):
        sequence, paso_ajuste = get_sequence(model_path, pad_target, iter_n, PASO_AJUSTE)
        pcas[i] = sequence_PCA(sequence, dims=2)
    
    model = model_path.split("/")[-3]
    plot_multiple_2D_pca(model, pcas)
    plot_multiple_3D_pca(model, pcas)


if __name__ == "__main__":
    model_path="5f_rgb_min/10000/10000.weights.h5"
    pad_target = load_target("data/Videos/heavy_diff_n=2.mp4")
    iter_n = 3_000
    PASO_AJUSTE = False
    k = 20
    
    get_metrics(model_path, pad_target, iter_n, PASO_AJUSTE)
    #multiple_PCA_plots(model_path, k, pad_target, iter_n, PASO_AJUSTE)
    
    
        
    