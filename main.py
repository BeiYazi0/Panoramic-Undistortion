import cv2
# import numpy as np
# import torch
import os
from utils import get_line, gif2png

from helper import TrainModelsTests, ProjectModelsTests
# from models import *
# from train import train_net

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    # test = ProjectModelsTests('./img/00000001.jpg')
    # test.testCylinderOrthogonal(160)
    # test.testSphericalOrthogonal(160)

    # test = TrainModelsTests('./img/00000001.jpg', './img/mask.jpg')
    # test.testSphericalDistortion(120, [0.55, 0.55], 0.5, 100, 0.005, True)

    # gif2png("./img/test1_2.gif")
    # gif2png("./img/test2.gif")
    # gif2png("./img/test3_1.gif")
    # gif2png("./img/test3_2.gif")

    test = TrainModelsTests('./img/301.jpg', './img/301_mask.jpg', './img/301_mask2.jpg')
    test.testSphericalDistortion(135, [0.55, 0.55], 0.5, 100, 0.005, True)
    get_line(cv2.imread('./img/301_mask2.jpg'))

    # height, width = srcImg.shape[:2]
    # sx, sy = 0.001, 0.001
    # window_size_f = 1.6
    #
    # binary, line_index = get_line(srcImg)
    #
    # net = RadialDistortion2(4, height, width, sx, sy, window_size_f)
    # train_net(net, 200, 6768, line_index, 0.0000003)
