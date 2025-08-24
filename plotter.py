import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy.typing import NDArray
import seaborn as sns
from sklearn.metrics import mean_squared_error

FloatArray = NDArray[np.float64]

class PlotData:
    """A class for handling plotting of regression data and predictions."""
    def __init__(self, figsize: Tuple[int, int] = (8, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.scatter_lim_x = (4, 25)
        self.scatter_lim_y = (-5, 25)
        
    def plot_line(self, X: FloatArray, y: FloatArray, p: FloatArray, 
                 use_ax: Tuple[int, int, int], title: str = "Training Data and Predicted Values",
                 xlabel: str = "Training Input", ylabel: str = "Target/Predicted") -> plt.Figure:
        ax = self.fig.add_subplot(*use_ax)
        ax.scatter(X, y)
        ax.plot(X, p, 'r')
        ax.grid(True)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return self.fig
    
    def plot_scatter(self, X: FloatArray, y: FloatArray, 
                    use_ax: Tuple[int, int, int], title: str = "Raw Data") -> plt.Figure:
        ax = self.fig.add_subplot(*use_ax)
        ax.scatter(X, y)
        ax.set(xlim=self.scatter_lim_x, ylim=self.scatter_lim_y)
        ax.grid(True)
        ax.set_title(title)
        return self.fig

    def plot_mse_vs_order(self, orders: List[int], mse_values: List[float],
                          xlabel: str = "Polynomial Order (m)", ylabel: str = "MSE",
                          title: str = "MSE vs. Polynomial Order") -> plt.Figure:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(orders, mse_values, "-o", color="royalblue", markerfacecolor='orange', label="MSE")
        ax.grid(True, ls='--', alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return fig

    def plot_grid(self, rows: int, cols: int, plots: List[Tuple[FloatArray, FloatArray, object, FloatArray, str]]) -> plt.Figure:
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = axs.flatten()
        for i, (x, y, poly_func, t, title) in enumerate(plots):
            axs[i].scatter(x, y, c='b', alpha=0.6, edgecolor='k', s=18, label="Data")
            axs[i].plot(t, poly_func(t), 'r-', lw=2, label=f"Fit (m={poly_func.order})")
            mse = mean_squared_error(y, poly_func(x))
            axs[i].set_title(f"{title}\nMSE={mse:.1f}")
            axs[i].grid(True, ls='--', alpha=0.5)
            axs[i].set_xlabel("x")
            axs[i].set_ylabel("y")
            axs[i].legend(fontsize=8)
        plt.tight_layout()
        return fig


def plot_greyscale(imgs):
    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(imgs[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(imgs[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(imgs[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(imgs[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

def plot_history(model, title=""):
    history_dict = model.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    epochs = range(1, len(acc_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

    plt.title('Training and validation loss ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_prediction_scatterplot(model, predicted, 
                                test_dtm, test_label, 
                                cluster_map, 
                                title="",
                                palette=["orange", "red", "blue", "green"]
    ):
    """ """

    X = model.fit_transform(test_dtm.toarray())

    # map colors for scatter plots
    pred_palette, gt_palette = [dict(zip(id, palette)) for id in zip(*cluster_map.items())]

    plt.figure(figsize=(16, 4))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1,2,1)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=test_label, palette=gt_palette)
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=predicted, palette=pred_palette)
    plt.title("Predicted")

    plt.show()
    
def show_semantic_network(G, title):
    """ show the network graph """

    plt.figure(figsize=(20,20), dpi=100)

    # node size proportional to eigenvector centrality of words
    node_size= [x*1000 for x in nx.eigenvector_centrality(G).values()]

    # choose a layout function
    pos=nx.kamada_kawai_layout(G)
    nx.drawing.nx_pylab.draw_networkx(G,
                                      node_size=node_size,
                                      node_color = "pink",
                                      pos=pos,
                                      edge_color='0.8',
                                      with_labels=True,
                                     font_size=12)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_mse(matrix_error):
    
    plt.figure(figsize=(10, 5))
    for i, errors in enumerate(matrix_error):
        plt.plot(errors, label=f'R{i+1} MSE')
    plt.xlabel('Step')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Matrix Factorization Error Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
