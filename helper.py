import io

import cv2
from IPython import display
from matplotlib import pyplot as plt
from PIL import Image

from utils import *
from models import *
from project_models import *
from train import train_net


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.history = []

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        plt.draw()
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

        buffer = io.BytesIO()
        self.fig.canvas.print_png(buffer)
        data = buffer.getvalue()
        buffer.write(data)
        img = Image.open(buffer)
        self.history.append(img)

    def save(self, file):
        self.history[0].save(file, save_all=True, append_images=self.history[1:], duration=50, loop=0)


class TrainModelsTests:
    def __init__(self, srcFile, *maskFile):
        # 读取原图和 mask
        self.srcImg = cv2.imread(srcFile)
        self.height, self.width = self.srcImg.shape[:2]

        self.line_index = []
        for mask in maskFile:
            mask = cv2.imread(mask)
            self.line_index.append(get_line(mask)[1])

    def testSphericalDistortion(self, angle, Dim, alpha, epochs=100, lr=0.005, show=False, file="img/test.gif"):
        net = SphericalDistortion(self.height, self.width, angle, Dim, alpha)
        projectModel = SphericalIsometric(self.srcImg)
        animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, epochs], ylim=[3.8, 4.5],
                            legend=['train loss'], ncols=2 if show else 1, figsize=(7, 2.5))
        train_net(net, projectModel, animator, epochs, self.line_index, self.width * self.height, lr, show)
        animator.save(file)
        # return net.undistort_parameters.detach()


class ProjectModelsTests:
    def __init__(self, srcFile):
        # 读取原图和 mask
        self.srcImg = cv2.imread(srcFile)
        self.height, self.width = self.srcImg.shape[:2]

    def testCylinderOrthogonal(self, angle, save='img/testCylinderOrthogonal.png'):
        projectModel = CylinderOrthogonal(self.srcImg)
        res = projectModel.undistort(angle)
        if save:
            cv2.imwrite(save, res)
        else:
            cv2.imshow("show", res)

    def testSphericalOrthogonal(self, angle, save='img/testSphericalOrthogonal.png'):
        projectModel = SphericalOrthogonal(self.srcImg)
        res = projectModel.undistort(angle)
        if save:
            cv2.imwrite(save, res)

    def testSphericalIsometric(self, angle, alpha=0.5, init_col=0, xDim=0, yDim=0, save='img/testSphericalIsometric.png'):
        projectModel = SphericalIsometric(self.srcImg)
        res = projectModel.undistort(angle, alpha, init_col, xDim, yDim)
        if save:
            cv2.imwrite(save, res)
