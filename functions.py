# coding: utf-8

import os
import sys
import shutil
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from image_processing import *
from constants import *


# Возвращение списка изображений в папке
def get_images_list():
    files_in_dir = os.listdir(IMG_DIR)
    images = list(filter(lambda f: f.endswith('.png'), files_in_dir))
    images = list(map(lambda x: IMG_DIR + x, images))
    # Перемешивание изображений
    random.seed(RAND_SEED)
    random.shuffle(images)
    return images


# Возвращает рандомный список капч и капч для тестирования (не включенных в 1 список)
def split_captcha_for_testing():
    main_list = get_images_list()
    if len(main_list) <= CAPTCHA_COUNT:
        raise Exception(f"Изображений должно быть больше {CAPTCHA_COUNT}!")
    n_captcha = main_list[-CAPTCHA_COUNT:]
    main_list = main_list[:-CAPTCHA_COUNT]
    print(f'Выделено {CAPTCHA_COUNT} капч для тестирования результата. '
          f'Для обучения и тестирования осталось {len(main_list)}.')
    return main_list, n_captcha


# Получить картинки всех букв из одной капчи
def get_letter_images_from_image(img_path):
    img_c1 = cv2.imread(img_path, cv2.CV_8UC1)
    img_otsu = image_to_otsu(img_c1)
    contours = get_countours(img_otsu, img_c1, False)
    letters_images = []
    for c in contours:
        x, y, w, h = c
        letter_img = img_otsu[y: y + h, x: x + w]
        letter_img = cv2.resize(letter_img, (WIDTH, WIDTH), interpolation=cv2.INTER_AREA)
        letters_images.append(letter_img)
    return letters_images


# Сохранить все изображения букв в соответствующие папки
def save_all_letters_to_dir(img_list, test=False):
    img_dir = 'test_letters' if test else 'letters'

    # Подтверждение
    inp = input('Действительно считать заново все буквы? '
                'Текущие данные будут удалены. Нажмите ENTER или n\n')
    if inp.lower() == 'n':
        print('Действие отменено!')
        return

    # Удаление текущих файлов
    try:
        if img_dir in os.listdir(os.getcwd()):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)
    except:
        print('Произошла ошибка при удалении файлов!')
        return
    print('Удаление файлов прошло успешно!')

    from collections import Counter
    counter = Counter()
    for img in img_list:
        img_name = img[-9:-4]
        letters_images = get_letter_images_from_image(img)
        for i, letter in enumerate(img_name):
            path = f'{img_dir}/{letter}/'
            if letter not in os.listdir(f'{img_dir}/'):
                os.mkdir(path)
            cv2.imwrite(f'{path}{letter}{counter[letter]}.png', letters_images[i])
            counter[letter] += 1
    print(f'{len(img_list)} изображений распознаны, и их буквы сохранены в соотствующие папки.')
    print(f'Количество сохранённых букв: {counter}')


# Возвращает рандомный список всех букв из папки letters
def get_all_letters_list(test=False):
    img_dir = 'test_letters/' if test else LETTERS_DIR
    letters = os.listdir(img_dir)
    letter_images_list = []
    for letter in letters:
        current_dir = img_dir + letter + '/'
        files_in_dir = os.listdir(current_dir)
        images = list(filter(lambda f: f.endswith('.png'), files_in_dir))
        images = list(map(lambda x: current_dir + x, images))
        letter_images_list.extend(images)
    # Перемешивание изображений
    random.seed(RAND_SEED)
    random.shuffle(letter_images_list)
    return letter_images_list


# Подготовить входные данные и метки для нейросети
def prepare_data_and_labels(width, test=False):
    data = []
    labels = []

    # Получаем список изображений
    image_paths = get_all_letters_list(test)

    for image_path in image_paths:
        # загружаем изображение, меняем размер на width x width пикселей (без учёта
        # соотношения сторон), сглаживаем его в width x width x 3 пикселей и
        # добавляем в список
        image = cv2.imread(image_path, cv2.CV_8UC1)  # cv2.imread(image_path)
        data.append(image)

        # извлекаем метку класса из пути к изображению и обновляем
        # список меток
        label = image_path.split('/')[-2]
        labels.append(label)

    # масштабируем интенсивности пикселей в диапазон [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


# Обработать и сохранить на диск 10 изображений
def process_n_images(images, n):
    for i in range(n):
        img_c1 = cv2.imread(images[i], cv2.CV_8UC1)
        img_otsu = image_to_otsu(img_c1)
        contours = get_countours(img_otsu, img_c1)
        result = contours_to_image(img_c1, contours)
        cv2.imwrite(f'{RESULT_DIR}{i}{IMG_FORMAT}', result)


# Подсчет кол-ва каждой буквы
def letters_counter():
    from collections import Counter
    counter = Counter()
    for im in os.listdir(IMG_DIR):
        for i in im[:-4]:
            counter[i] += 1
    print(counter)


def print_results_of_neural_network(model, results):
    print()
    print(
        f'Val_loss in different epochs={[(ep, round(value, 2)) for ep, value in enumerate(results.history["val_loss"])]}')
    print(
        f'Val_accuracy in different epochs={[(ep, round(value, 2)) for ep, value in enumerate(results.history["val_accuracy"])]}')
    print("Mean Test-Accuracy:", np.mean(results.history["val_accuracy"]))
    captcha_recognition_result = [f"{x[0]}={x[1]}/{CAPTCHA_COUNT}, {x[2]}/{x[3]}" for x in model.captcha_result]
    print(f'Результат распознавания капч по эпохам: {captcha_recognition_result}')
    results.history['correct captha'] = [x[1] / CAPTCHA_COUNT for x in model.captcha_result]
    results.history['correct letters'] = [x[2] / x[3] for x in model.captcha_result]
    print()
