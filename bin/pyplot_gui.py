import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np

from helpers import Skeleton
from parameters import Parameters

class PyplotWindow(tk.Frame):
    def __init__(self, root:tk.Tk, params:Parameters, VR_skeleton:Skeleton, *args, **kwargs):
        self.root = root
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.SinBool = True
        self.params = params
        self.VR_skeleton = VR_skeleton

        self.PlotButton = tk.Button(self.root, text='Plot', command=self.start_plot)
        self.PlotButton.pack()
        self.UpdateButton = tk.Button(self.root, text='Update', command=self.update_plot)
        self.UpdateButton.pack()

    def start_plot(self):
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(projection = "3d")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.canvas = FigureCanvasTkAgg(self.fig, master = self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.PlotButton['state'] = tk.DISABLED

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        for key, value in self.VR_skeleton.skeleton_points.items():
            self.ax.scatter(self.VR_skeleton.skeleton_points[key][0], 
                            self.VR_skeleton.skeleton_points[key][2],
                            self.VR_skeleton.skeleton_points[key][1], 
                            marker = 'o',
                            color = 'red')

            self.ax.scatter(self.params.VR_skeleton_og.skeleton_points[key][0], 
                            self.params.VR_skeleton_og.skeleton_points[key][2],
                            self.params.VR_skeleton_og.skeleton_points[key][1], 
                            marker = '^',
                            color = "blue")

    def _quit(self):
        self.root.quit()
        self.root.destroy()


def make_pyplot_gui(params:Parameters, VR_skeleton:Skeleton):
    root = tk.Tk()
    root.wm_title("Plot embedded in tk")
    PyplotWindow(root, params, VR_skeleton).pack(side="top", fill="both", expand=True)
    root.mainloop()