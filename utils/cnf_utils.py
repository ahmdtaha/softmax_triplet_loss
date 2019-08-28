import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools


def cnf_accuracy(cnf_matrix):
    return np.trace(cnf_matrix) / np.sum(cnf_matrix)

def cnf_mean_class_accuracy(cnf_matrix,num_classes):
    return np.trace(cnf_matrix / cnf_matrix.sum(axis=1)[:, None]) / num_classes


def plot_confusion_matrix(cm, classes=[],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,fmt='d'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')