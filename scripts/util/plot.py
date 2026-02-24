import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
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

class DoubleLinePlot(Plot):
    def __init__(self, title, xlabel, ylabel, line_label1, line_label2, ion = True, figsize = (8, 5), line_color1 = "blue", line_color2 = "orange"):
        super().__init__(title, xlabel, ylabel, line_label1, ion, figsize, line_color1)
        self.line2, = self.ax.plot([], [], label=line_label2, color=f"tab:{line_color2}")
        self.line.set_linewidth(2)
        self.line2.set_linewidth(2)
        self.ax.legend()

    def update_with_x(self, data, data2, x, y_lim=None):
        self.line2.set_data(x, data2)
        super().update_with_x(data, x)
        if y_lim != None:
            self.ax.set_ylim(bottom=y_lim)  # top=self.ax.get_ylim()[1] * 1.05
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update(self, data, data2):
        x = np.arange(len(data))
        self.update_with_x(data, data2, x)


class FiveLinePlot(DoubleLinePlot):
    def __init__(self, title, xlabel, ylabel, line_labels: list, ion=True, figsize=(8, 5), line_colors = ["green", "olive", "orange", "red", "brown"]):
        super().__init__(title, xlabel, ylabel, line_labels[0], line_labels[1], ion, figsize, line_colors[0], line_colors[1])
        self.line3, = self.ax.plot([], [], label=line_labels[2], color=f"tab:{line_colors[2]}")
        self.line4, = self.ax.plot([], [], label=line_labels[3], color=f"tab:{line_colors[3]}")
        self.line5, = self.ax.plot([], [], label=line_labels[4], color=f"tab:{line_colors[4]}")
        self.line3.set_linewidth(2)
        self.line4.set_linewidth(2)
        self.line5.set_linewidth(2)
        self.ax.legend()

    def update_with_x(self, data, data2, data3, data4, data5, x, y_lim=None):
        self.line3.set_data(x, data3)
        self.line4.set_data(x, data4)
        self.line5.set_data(x, data5)
        super().update_with_x(data, data2, x)
        if y_lim != None:
            self.ax.set_ylim(bottom=y_lim)  # top=self.ax.get_ylim()[1] * 1.05
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update(self, data, data2, data3, data4, data5, y_lim):
        x = np.arange(len(data))
        self.update_with_x(data, data2, data3, data4, data5, x, y_lim)

    
class LogPlot(Plot):
    def __init__(self, title, xlabel, ylabel, line_label, ion = True, figsize = (8, 5), line_color = "blue"):
        super().__init__(title, xlabel, ylabel, line_label, ion, figsize, line_color)

    def update_with_x(self, data, x):
        # self.ax.set_xticks(x)
        # self.ax.set_xticklabels([str(v) for v in x])
        # self.ax.set_xscale('log')
        if not self.ax.get_xscale() == 'log':
            self.ax.set_xlim(min(x), max(x))
            self.ax.set_xscale('log')
        self.ax.xaxis.set_major_locator(FixedLocator(x))
        self.ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x]))
        return super().update_with_x(data, x)
    
class DoubleLogPlot(DoubleLinePlot):
    def __init__(self, title, xlabel, ylabel, line_label1, line_label2, ion=True, figsize=(8, 5), line_color1="blue", line_color2="orange"):
        super().__init__(title, xlabel, ylabel, line_label1, line_label2, ion, figsize, line_color1, line_color2)

    def update_with_x(self, data, data2, x, y_lim=None):
        if not self.ax.get_xscale() == 'log':
            self.ax.set_xlim(min(x), max(x))
            self.ax.set_xscale('log')
        self.ax.xaxis.set_major_locator(FixedLocator(x))
        self.ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x]))
        return super().update_with_x(data, data2, x, y_lim)