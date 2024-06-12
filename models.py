import torch
from torch import nn


class RadialDistortion(nn.Module):
    def __init__(self, feature_dim, height, width, sx, sy, window_size_f):
        super().__init__()
        self.feature_dim = feature_dim
        self.s = torch.tensor((sx, sy))
        self.window_size_f = window_size_f
        self.img_size = torch.tensor((width, height)) * window_size_f
        self.center = self.img_size / 2

        k = torch.randn(feature_dim, )
        k[0] = -0.6
        k[1] = 0.2
        self.k = nn.Parameter(k)

    def forward(self, X):
        """

        :param X [m, 2]: input
        :return Y [m, 1]: ouput
        """
        m = X.shape[0]

        x_corr = (X - self.center)
        x_corr *= self.s
        feature = torch.sum(x_corr ** 2, dim=1)

        factor = torch.ones((m,))
        for i in range(self.feature_dim):
            factor += self.k[i] * feature ** (i + 1)
        Y = x_corr[:, 1] * factor
        return Y / self.s[1] + self.center[1] / self.window_size_f


class RadialDistortion2(nn.Module):
    def __init__(self, feature_dim, height, width, sx, sy, window_size_f):
        super().__init__()
        self.feature_dim = feature_dim
        self.s = torch.tensor((sx, sy))
        self.window_size_f = window_size_f
        self.img_size = torch.tensor((width, height)) * window_size_f
        self.center = self.img_size / 2

        k = torch.tensor([-0.7784, 0.3743, 0.3620, -0.1719])
        self.k = nn.Parameter(k)
        self.p = nn.Parameter(torch.tensor([0.0492, 0.2349]))

    def forward(self, X):
        """

        :param X [m, 2]: input
        :return Y [m, 1]: ouput
        """
        m = X.shape[0]

        x_corr = (X - self.center)
        x_corr *= self.s
        feature = torch.sum(x_corr ** 2, dim=1)

        factor = torch.ones((m,))
        for i in range(self.feature_dim):
            factor += self.k[i] * feature ** (i + 1)
        print(factor)
        Y = x_corr[:, 1] * factor
        Y += 2 * self.p[1] * torch.prod(x_corr, dim=1) + self.p[0] * (feature + 2 * x_corr[:, 1] ** 2)
        return Y / self.s[1] + self.center[1] / self.window_size_f


class SphericalDistortion(nn.Module):
    def __init__(self, height, width, angle, Dim, alpha):
        super().__init__()
        self.img_size = torch.tensor((width, height))
        self.Dim = torch.tensor(Dim)
        self.undistort_parameters = nn.Parameter(torch.tensor([angle/180, alpha, angle/180]))
        self.s = torch.sqrt(torch.tensor(height * width))
        self.z = 0

    def get_penalty(self):
        z = self.z
        alpha = self.undistort_parameters[1]
        r = - torch.log(z / self.s) - 1.*(torch.log(1 - alpha) + torch.log(alpha))
        return r

    def forward(self, X):
        """

        :param X [m, 2]: input
        :return Y [m, 1]: ouput
        """
        angle, alpha, init_col = self.undistort_parameters
        offset_w = init_col * torch.pi / 2

        p = max(alpha, 1 - alpha)
        r = self.img_size[0] / (angle * torch.pi)
        xDim, yDim = self.Dim * self.img_size
        z = torch.sqrt(r ** 2 - xDim * xDim / 4.0 - yDim * yDim * p * p)  # 球心到平面的距离
        self.z = z

        x = xDim / 2 + z * torch.tan(X[:, 0] / r - offset_w)
        y = yDim * alpha + torch.sqrt((x - xDim / 2)**2 + z**2) * torch.tan((X[:, 1] - self.img_size[1] * alpha) / r)

        return y
