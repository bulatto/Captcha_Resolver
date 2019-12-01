# coding: utf-8

import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from functions import *


# img_list, captcha_list = split_captcha_for_testing()
# save_all_letters_to_dir(img_list)

process_n_images(get_images_list(), 100)

# Получение списка картинок
# images = get_images_list()
# get_images_before_and_after_contouring(images[67])


# Проверить капчу
# img_list, captcha_list = split_captcha_for_testing()
# c = 0
# for i in captcha_list:
#     if c<16:
#         c+=1
#         continue
#     get_images_before_and_after_contouring(i)
