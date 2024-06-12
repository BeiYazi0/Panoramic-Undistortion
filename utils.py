import cv2
import numpy as np

from PIL import Image
import os

from matplotlib_inline import backend_inline


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def get_line(srcImg):
    # 灰度化
    gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

    # 二值化
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./img/m.jpg", binary)

    y, x = np.where(binary > 0)
    line_index = np.concatenate((x, y), axis=0).reshape(2, -1).T
    return binary, line_index


def gif2png(gif_file):
    im = Image.open(gif_file)
    pngDir = gif_file[:-4]
    # 创建存放每帧图片的文件夹(文件夹名与图片名称相同)
    os.mkdir(pngDir)
    try:
        while True:
            # 保存当前帧图片
            current = im.tell()
            im.save(pngDir + '/' + str(current) + '.png')
            # 获取下一帧图片
            im.seek(current + 1)
    except EOFError:
        pass