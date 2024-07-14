import tkinter as tk
from tkinter import filedialog, simpledialog

import cv2
import numpy
from PIL import Image, ImageTk

from project_models import SphericalIsometric


class GUI:
    def __init__(self, master, width=1600, height=800):
        self.master = master
        self.master.title("Image Correction GUI")

        ## 矫正模型
        self.projectModel = None

        # 左侧Canvas
        self.canvas = tk.Canvas(master, width=width, height=height, bg='white')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # 右侧文本框和按钮
        child_window = tk.Toplevel()
        child_window.title("operation")

        frame = tk.Frame(child_window)
        frame.grid(row=0, column=1, sticky="nsew")

        # 创建文本框和按钮
        self.variables = {k: tk.StringVar(value='0.75') for k in ['angle', 'alpha', 'init_col', 'dimX', 'dimY']}
        self.textboxes = {}
        for i, (name, var) in enumerate(self.variables.items()):
            label = tk.Label(frame, text=name)
            label.grid(row=i, column=0)
            textbox = tk.Entry(frame, textvariable=var)
            textbox.grid(row=i, column=1)
            btn_dec = tk.Button(frame, text="−", command=lambda k=name: self.change_value(k, -0.005))
            btn_dec.grid(row=i, column=2)
            btn_inc = tk.Button(frame, text="+", command=lambda k=name: self.change_value(k, +0.005))
            btn_inc.grid(row=i, column=3)
            self.textboxes[name] = (textbox, var)

        # 右上方按钮
        btn_load = tk.Button(frame, text="加载图像", command=self.load_image)
        btn_load.grid(row=i+1, column=0, sticky="w")
        btn_correct = tk.Button(frame, text="矫正图像", command=self.correct_image)
        btn_correct.grid(row=i+2, column=0, sticky="w")
        btn_save = tk.Button(frame, text="导出矫正配置", command=self.save_settings)
        btn_save.grid(row=i + 3, column=0, sticky="w")

        # 设置主窗口扩展性
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        def disable_event():
            pass
        child_window.protocol("WM_DELETE_WINDOW", disable_event)
        child_window.mainloop()

    def change_value(self, name, delta=0.001, max_value=1):
        var = self.variables[name]
        value = float(var.get())
        value += delta
        if value < 0:
            value = 0
        elif value > max_value:
            value = max_value
        var.set(str(round(value, 3)))
        if self.projectModel is not None:
            self.correct_image()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.canvas.delete("all")
            img = Image.open(file_path)
            self.variables['alpha'].set(0.5)
            self.variables['dimX'].set(0.55)
            self.variables['dimY'].set(0.55)
            self.projectModel = SphericalIsometric(cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR))

            img_tk = ImageTk.PhotoImage(img.resize((self.canvas.winfo_width(), self.canvas.winfo_height())))
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.canvas.image = img_tk

    def correct_image(self):
        if self.projectModel is None:
            return
        settings = [float(self.textboxes[k][1].get()) for k in self.variables]
        img = self.projectModel.undistort(*settings)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_tk = ImageTk.PhotoImage(img.resize((self.canvas.winfo_width(), self.canvas.winfo_height())))
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

    def save_settings(self, file="undistort_setting.txt"):
        if self.projectModel is None:
            return
        settings = [float(self.textboxes[k][1].get()) for k in self.variables]
        settings[0] = int(settings[0] * 180)
        settings[1] = int(settings[1] * 180)
        settings[2] = int(settings[2] * 180)

        w, h = self.projectModel.img_size
        settings[-2] = int(settings[-2] * w)
        settings[-1] = int(settings[-1] * h)

        with open(file, encoding="utf-8", mode='w') as f:
            f.write(" ".join([str(val) for val in settings]))


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x800+20+40")
    app = GUI(root)
    root.mainloop()