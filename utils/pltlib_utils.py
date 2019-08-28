import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
def imshow_annotated(mat,save_path,title):
    plt.imshow(mat, interpolation='nearest')
    plt.title(title)
    # plt.colorbar()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = plt.text(j, i, np.round(mat[i, j], 2),
                            ha="center", va="center", color="w")

    plt.savefig(save_path)
    plt.close()