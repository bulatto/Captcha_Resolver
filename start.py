# coding: utf-8

import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

from functions import *


# img_list, captcha_list = split_captcha_for_testing()
# save_all_letters_to_dir(captcha_list, True)


# process_n_images(get_images_list(), 160)

# Получение списка картинок
# images = get_images_list()
# get_images_before_and_after_contouring(images[67])


# Проверить капчу
img_list, captcha_list = split_captcha_for_testing()
for captcha in captcha_list:
    trainX, trainY = prepare_data_and_labels(WIDTH)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)

    cdata = np.array(get_letter_images_from_image(captcha), dtype="float") / 255.0
    cdata = cdata.reshape(-1, WIDTH, WIDTH, 1)
    clabel = np.array(captcha.split('/')[-1][:-4])
    model = load_model('output/my_model.h5')
    predictions = model.predict(cdata, batch_size=32)
    out_labels = lb.inverse_transform(predictions)
    print('------------------------------')
    print(out_labels)
    print('------------------------------')
    get_images_before_and_after_contouring(captcha)
