# coding: utf-8

import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from functions import *


# Получение списка картинок
images = get_images_list()
img_c1 = cv2.imread(images[34], cv2.CV_8UC1)
img_otsu = image_to_otsu(img_c1)
contours = get_countours(img_otsu, img_c1)
# Накладывание контуров на изображение
result = contours_to_image(img_c1, contours)
# Сравнение изображений
plt_sub(1, 2, [img_c1, result], ['После обработки', 'Результат'])
