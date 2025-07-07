import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy.typing import NDArray
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
