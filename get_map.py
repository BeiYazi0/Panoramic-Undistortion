import cv2
import numpy as np
from typing import List


# 全景图width方向没有变形，沿width方向等比例放缩
def change_w(x: np.ndarray, y: np.ndarray, xDim: float, yDim: float, z: float, r: float, init_col: float) -> np.ndarray:
    tt:np.ndarray = (xDim - x) / z
    l:np.ndarray = init_col * np.pi / 360 - np.arctan(tt)
    result:np.ndarray = l * r
    return result

# 全景图height方向有变形，沿height方向矫正
def change_h(x: np.ndarray, y: np.ndarray, xDim: float, yDim: float, z: float, r: float, height: float, alpha:float) -> np.ndarray:
    tt:np.ndarray = (y - yDim) / np.sqrt(np.power(x - xDim, 2) + z * z)
    l:np.ndarray = np.arctan(tt)
    result:np.ndarray = l * r + height * alpha
    return result


def map_create(width, height, angle, alpha = 0.5, init_col = 0, xDim = 0, yDim = 0):
    # angle 张角(单位°，沿width方向)
    print(width, height)

    p = max(alpha, 1- alpha)

    # 平面图大小
    if yDim == 0:
        yDim = int(height * 0.6)

    if xDim == 0:
        xDim = int(width * 0.6)
    
    r = width / (angle / 180 * np.pi)
    z = np.sqrt(r*r - xDim*xDim/4.0 - yDim*yDim*p*p) # 球心到平面的距离
    if init_col == 0:
        init_col = angle * 1.04
    
    # 生成索引
    x, y = np.meshgrid(np.arange(xDim), np.arange(yDim))
    # 对索引进行变换
    row:np.ndarray = change_h(x, y, xDim / 2.0, yDim * alpha, z, r, height, alpha).astype(np.float32)
    col:np.ndarray = change_w(x, y, xDim / 2.0, yDim * alpha, z, r, init_col).astype(np.float32) 

    return col, row
