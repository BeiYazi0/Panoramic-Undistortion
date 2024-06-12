import numpy as np
import cv2

from utils import get_line


class ProjectModel:
    """投影模型基本接口"""
    def __init__(self, img):
        self.src, self.img_size, self.r = self._cut(img)
        self.u0, self.v0 = self.img_size[0] // 2, self.img_size[1] // 2

    # 有效区域截取
    def _cut(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(cnts)
        r = max(w / 2, h / 2)
        # 提取有效区域
        img_valid = img[y:y + h, x:x + w]
        return img_valid, (w, h), int(r)

    def map_create(self, *args):
        raise NotImplementedError

    def undistort(self, *args):
        col, row = self.map_create(*args)
        res = cv2.remap(self.src, col, row, cv2.INTER_LINEAR)
        return res


class CylinderOrthogonal(ProjectModel):
    def __init__(self, img):
        super(CylinderOrthogonal, self).__init__(img)

    def map_create(self, angle):
        width, height = self.img_size

        # angle 张角(单位°，沿width方向)
        r = height / (angle / 180 * np.pi)

        # 平面图大小
        yDim = int(2 * r * np.sin(height / 2 / r))
        xDim = width

        # 生成索引
        x, y = np.meshgrid(np.arange(xDim), np.arange(yDim))
        # 对索引进行变换
        col = x.astype(np.float32)

        y += int(r - yDim/2)
        c = int(np.pi * r / 2 - height / 2)
        row = 2 * r * np.arctan(1 / np.sqrt(2 * r / y - 1)) - c
        return col, row.astype(np.float32)


class SphericalOrthogonal(ProjectModel):
    def __init__(self, img):
        super(SphericalOrthogonal, self).__init__(img)

    def map_create(self, angle):
        width, height = self.img_size

        # angle 张角(单位°，沿width方向)
        r = width / (angle / 180 * np.pi)

        # 平面图大小
        yDim = int(2 * r * np.sin(height / 2 / r))
        xDim = int(2 * r * np.sin(width / 2 / r))

        # 生成索引
        x, y = np.meshgrid(np.arange(xDim), np.arange(yDim))
        # 对索引进行变换(将x和y调成对称，等价于x-r, y-r)
        x -= xDim // 2
        y -= yDim // 2

        ry = np.sqrt(r ** 2 - y ** 2)
        col = ry * np.arcsin(x / ry) + width // 2

        rx = np.sqrt(r ** 2 - x ** 2)
        row = rx * np.arcsin(y / rx) + height // 2
        return col.astype(np.float32), row.astype(np.float32)


class DoubleLongitude(ProjectModel):
    def __init__(self, img):
        super(DoubleLongitude, self).__init__(img)

    # 计算光学中心与球面半径
    def _get_center_L(self, masks):
        def _get_oval_paraments(x, y):
            assert x.shape == y.shape
            N = x.shape[0]
            x2, x3 = x ** 2, x ** 3
            y2, y3, y4 = y ** 2, y ** 3, y ** 4
            xy = x * y
            P = np.array([[np.sum(xy ** 2), np.sum(x * y3), np.sum(x2 * y), np.sum(x * y2), np.sum(xy)],
                          [np.sum(x * y3), np.sum(y4), np.sum(x * y2), np.sum(y3), np.sum(y2)],
                          [np.sum(x2 * y), np.sum(x * y2), np.sum(x2), np.sum(xy), np.sum(x)],
                          [np.sum(x * y2), np.sum(y3), np.sum(xy), np.sum(y2), np.sum(y)],
                          [np.sum(xy), np.sum(y2), np.sum(x), np.sum(y), N]])

            Q = np.array([np.sum(x3 * y), np.sum(xy ** 2), np.sum(x3), np.sum(x2 * y), np.sum(x2)]).reshape((-1, 1))

            param = -np.linalg.inv(P) @ Q
            return param

        def _get_per_parms(mask):
            _, line_index = get_line(mask)
            A, B, C, D, E = _get_oval_paraments(line_index[0], line_index[1])
            term = A ** 2 - 4 * B

            u0 = (2 * B * C - A * D) / term
            v0 = (2 * D - A * C) / term
            p1 = (B * (C ** 2) - A * C * D + D ** 2) / term + E
            p2 = 2 * (1 + B + np.sqrt((1 - B) ** 2 + A ** 2)) / term
            L = np.sqrt(p1 * p2)
            return np.array([u0, v0, L])

        res = np.zeros(3)
        for mask in masks:
            res += _get_per_parms(mask)
        return res / len(masks)

    def set_masks(self, masks):
        self.u0, self.v0, self.r = self._get_center_L(masks)

    def map_create(self, mode='Orthogonal'):
        if mode == 'Orthogonal':
            return self.undistortOrthogonal()
        elif mode == 'Isometric':
            return self.undistortIsometric()
        else:
            raise Exception("Invalid mode name!")

    # 鱼眼矫正
    def undistortOrthogonal(self):
        src, r = self.src, self.r
        # r： 半径， R: 直径
        R = 2 * r
        # Pi: 圆周率
        Pi = np.pi
        # 存储映射结果
        dst = np.zeros((R, R, 3))
        src_h, src_w, _ = src.shape

        # 光学中心
        x0, y0 = self.u0, self.v0

        # 数组， 循环每个点
        range_arr = np.array([range(R)])

        theta = Pi - (Pi / R) * (range_arr.T)
        temp_theta = np.tan(theta) ** 2

        phi = Pi - (Pi / R) * range_arr
        temp_phi = np.tan(phi) ** 2

        tempu = r / (temp_phi + 1 + temp_phi / temp_theta) ** 0.5
        tempv = r / (temp_theta + 1 + temp_theta / temp_phi) ** 0.5

        # 用于修正正负号
        flag = np.array([-1] * r + [1] * r)

        # 加0.5是为了四舍五入求最近点
        u = x0 + tempu * flag + 0.5
        v = y0 + tempv * np.array([flag]).T + 0.5

        # 插值
        dst = cv2.remap(src, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
        return dst

    def undistortIsometric(self):
        src, r = self.src, self.L
        # r： 半径， R: 直径
        R = 2 * r
        # Pi: 圆周率
        Pi = np.pi
        # 存储映射结果
        dst = np.zeros((R, R, 3))
        src_h, src_w, _ = src.shape

        # 光学中心
        x0, y0 = self.u0, self.v0

        # 数组， 循环每个点
        range_arr = np.array([range(R)])

        theta = Pi - (Pi / R) * (range_arr.T)
        temp_theta = np.tan(theta) ** 2

        phi = Pi - (Pi / R) * range_arr
        temp_phi = np.tan(phi) ** 2

        # 用于修正正负号
        flag = np.array([-1] * r + [1] * r)

        x = r / (temp_phi + 1 + temp_phi / temp_theta) ** 0.5 * flag
        y = r / (temp_theta + 1 + temp_theta / temp_phi) ** 0.5 * np.array([flag]).T
        z_r = 1 / (1 + 1 / temp_theta + 1 / temp_phi) ** 0.5

        #
        w = np.arccos(z_r)

        # 加0.5是为了四舍五入求最近点

        u = x0 + r * w * x / ((x ** 2 + y ** 2) ** 0.5)
        v = y0 + r * w * y / ((x ** 2 + y ** 2) ** 0.5)

        # 插值
        dst = cv2.remap(src, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
        return dst


class SphericalIsometric(ProjectModel):
    def __init__(self, img):
        super(SphericalIsometric, self).__init__(img)

    def map_create(self, angle, alpha=0.5, init_col=0, xDim=0, yDim=0):
        width, height = self.img_size

        # angle 张角(单位°，沿width方向)
        p = max(alpha, 1 - alpha)
        # 平面图大小
        if yDim == 0:
            yDim = int(height * 0.6)
        else:
            yDim = int(height * yDim)
        if xDim == 0:
            xDim = int(width * 0.6)
        else:
            xDim = int(width * xDim)

        r = width / (angle * np.pi)
        z = np.sqrt(r * r - xDim * xDim / 4.0 - yDim * yDim * p * p)  # 球心到平面的距离

        def change_w(x):
            tt = (xDim/2 - x) / z
            l = init_col * np.pi / 2 - np.arctan(tt)  # angle/2 + a
            result = l * r
            return result

        # 全景图height方向有变形，沿height方向矫正
        def change_h(x, y):
            tt = (y - yDim * alpha) / np.sqrt(np.power(x - xDim/2, 2) + z * z)
            l = np.arctan(tt)
            result = l * r + height * alpha
            return result

        # 生成索引
        x, y = np.meshgrid(np.arange(xDim), np.arange(yDim))
        # 对索引进行变换
        row = change_h(x, y).astype(np.float32)
        col = change_w(x).astype(np.float32)

        return col, row
