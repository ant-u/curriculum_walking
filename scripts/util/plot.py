import matplotlib.pyplot as plt
import numpy as np

class Plot():
    """Class for creating a single line plot"""
    def __init__(self, 
            title: str, xlabel: str, 
            ylabel: str, line_label: str,
            ion: bool = True, figsize: tuple = (8,5),
            line_color: str = "blue"
        ):
        if ion:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.line, = self.ax.plot([], [], label=line_label, color=f"tab:{line_color}")
        self.fill = None
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.fig.tight_layout()

    def update_with_x(self, data, x):
        """update plot with given data and x, must match in dimensions"""
        self.line.set_data(x, data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, data):
        """update plot with given data, arranged length is taken as x data"""
        x = np.arange(len(data))
        self.update_with_x(data, x)        

    def save(self, path: str):
        """Save at given path"""
        self.fig.savefig(path)


class IQRPlot(Plot):
    """Creating a plot with single line but also quantile ranges"""
    def __init__(self, 
            title: str, xlabel: str, 
            ylabel: str, line_label: str,
            ion: bool = True, figsize: tuple = (8,5),
            line_color: str = "blue"
        ):
        super().__init__(title, xlabel, ylabel, line_label, ion, figsize, line_color)
        self.line_color = line_color
        self.fill = None

    def update(self, data, q_lower, q_upper):
        """update plot with data. Every call must provide all data to be plotted.
        Also data, q_lower and q_upper must have same lenght (for mapping to x axis)."""
        x = np.arange(len(data))
        self.update_with_x(data, x, q_lower, q_upper)

    def update_with_x(self, data, x, q_lower, q_upper):
        """Same as update, just with custom x field"""
        self.line.set_data(x, data)
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.ax.fill_between(x, q_lower, q_upper, color=self.line_color, alpha=0.2, label="IQR")
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()