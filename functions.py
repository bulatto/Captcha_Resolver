# coding: utf-8

import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_dir = 'captcha_dataset/samples/'
letters_dir = 'letters/'
result_dir = 'results/'
img_format = '.png'
rand_seed = 2


# Вывод изображения в отдельном окне
def cv2_show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Вывод изображения в notebook
def plt_image(t, image):
    plt.figure()
    plt.title(t)
    plt.imshow(image, 'gray')
    plt.show()


# Вывод изображения в виде сетки
def plt_sub(m, n, img_list, titles_list=None):
    for i, img in enumerate(img_list):
        plt.subplot(m, n, i + 1)
        plt.imshow(img, 'gray')
        if titles_list is not None:
            plt.title(titles_list[i])
        else:
            plt.title(str(i + 1))
    plt.show()


# Возвращает рандомный список всех букв
def get_all_letters_list():
    letters = os.listdir(letters_dir)
    letter_images_list = []
    for letter in letters:
        current_dir = letters_dir + letter + '/'
        files_in_dir = os.listdir(current_dir)
        images = list(filter(lambda f: f.endswith('.png'), files_in_dir))
        images = list(map(lambda x: current_dir + x, images))
        letter_images_list.extend(images)
    # Перемешивание изображений
    random.seed(rand_seed)
    random.shuffle(letter_images_list)
    return letter_images_list


# Возвращение списка изображений в папке
def get_images_list():
    files_in_dir = os.listdir(img_dir)
    images = list(filter(lambda f: f.endswith('.png'), files_in_dir))
    images = list(map(lambda x: img_dir + x, images))
    # Перемешивание изображений
    random.seed(rand_seed)
    random.shuffle(images)
    return images


# Обработать и сохранить на диск 10 изображений
def process_n_images(images, n):
    for i in range(n):
        img_c1 = cv2.imread(images[i], cv2.CV_8UC1)
        img_otsu = image_to_otsu(img_c1)
        contours = get_countours(img_otsu, img_c1)
        result = contours_to_image(img_c1, contours)
        cv2.imwrite(f'{result_dir}{i}{img_format}', result)


# Сохранить изображения определённой буквы в файлы
def save_some_letter(letter):
    if len(letter) != 1:
        return
    path = f'letters/{letter}/'
    if letter not in os.listdir('letters/'):
        os.mkdir(path)
    images = [im for im in get_images_list() if letter in im[len(img_dir):-4]]
    img_count = 0
    for i,image in enumerate(images):
        name = image[len(img_dir):-4]
        positions = [i for i, let in enumerate(name) if let == letter]
        for pos in positions:
            img_c1 = cv2.imread(image, cv2.CV_8UC1)
            img_otsu = image_to_otsu(img_c1)
            contours = get_countours(img_otsu, img_c1)
            if pos < len(contours):
                x, y, w, h = contours[pos]
                letter_img = img_otsu[y: y + h, x: x + w]
                letter_img = cv2.resize(letter_img, (30, 30), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f'{path}{letter}{i}.png', letter_img)
                img_count += 1

# Подсчет кол-ва каждой буквы
def letters_counter():
    from collections import Counter
    counter = Counter()
    for im in os.listdir(img_dir):
        for i in im[:-4]:
            counter[i] += 1
    print(sorted(counter))


# Otsu thresholding (CV_8UC1 only)
def image_to_otsu(img_c1):
    ret2, th2 = cv2.threshold(img_c1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation
    kernel = np.ones((3, 3), np.uint8)
    dilation1 = cv2.dilate(th2, kernel, iterations=1)
    # erosion
    erosion = cv2.erode(dilation1, kernel, iterations=1)
    # dilation
    kernel = np.ones((3, 1), np.uint8)
    dilation2 = cv2.dilate(erosion, kernel, iterations=1)
    # plt_sub(2,2,[erosion, dilation2])  # Вывод 2 изображений для сравнения
    return dilation2


# Инверсия цвета
# cv2.bitwise_not(image)

# Нарисовать все контуры по очереди
# for i in range(len(contours)):
#     my = original.copy()
#     cv2.drawContours(my, contours, i, (255, 0, 0), 1)
#     plt_image('my', my)


# Обработка контуров
def contour_processing(contrs):
    was_change = True

    # Соединение
    def joining(i, j):
        ix, iy, iw, ih = contrs[i]
        jx, jy, jw, jh = contrs[j]

        # Условия, что контуры маленькие и близкие
        cases = [abs(ix - jx) < 15, iw * ih + jw * jh < 600]

        # Если один контур внутри другого
        is_in = ix <= jx <= ix + iw and (ix + ih - jx) / jw >= 0.5

        if all(cases) or is_in:
            print('Соединение контуров ...')
            x, y = min(ix, jx), min(iy, jy)
            w = max(ix + iw - x, jx + jw - x)
            h = max(iy + ih - y, jy + jh - y)
            if w < 20 * 2 and h < 20 *2:
                contrs[i] = (x, y, w, h)
                del contrs[j]
                return True
        return False

    # Разъединение контуров
    def separaton(i):
        ix, iy, iw, ih = contrs[i]
        # Если контур включает несколько знаков
        k = iw // 20
        if k > 1:
            print(f'Разъединение контуров (k={k})...')
            if i == 0:
                dx = (iw - 5) // k
                contrs[i] = (ix, iy, dx + 5, ih)
                x_now = ix + dx + 5
            else:
                dx = iw // k
                contrs[i] = (ix, iy, dx, ih)
                x_now = ix + dx
            for z in range(k - 1):
                contrs.append((x_now, iy, dx, ih))
                x_now += dx
            return True
        return False

    while was_change:
        was_change = False
        contrs = sorted(contrs)
        length = len(contrs)
        for i in range(length):
            # Разделение
            was_change = separaton(i)
            if was_change:
                break

            for j in range(i + 1, length):
                # Соединение
                was_change = joining(i, j)
                if was_change:
                    break
            if was_change:
                break

    # Удаление ненужных контуров
    contrs = [c for c in contrs if 100 <= c[2] * c[3] < 200 * 50]

    # Если перебор букв
    while len(contrs) > 5:
        sizes = [c[2] * c[3] for c in contrs]
        min_size = min(sizes)
        ind = [i for i, c in enumerate(contrs) if c[2] * c[3] == min_size][0]
        del contrs[ind]

    # Если не хватает букв
    if len(contrs) == 4:
        print("Букв не хватает. Делим один контур пополам ...")
        w = [c[2] for c in contrs]
        w_max_ind = [i for i, c in enumerate(contrs) if c[2] == max(w)][0]

        dx = contrs[w_max_ind][2] // 2
        zx, zy, zw, zh = contrs[w_max_ind]
        contrs[w_max_ind] = (zx, zy, dx, zh)
        contrs.append((zx + dx, zy, dx, zh))
    return contrs


# Накладывание контуров на изображение
def contours_to_image(image, contrs):
    output = image.copy()
    for i in contrs:
        x, y, w, h = i
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return output


# Вывод исходного изображения и рядом все его контуры
def plt_image_with_all_contours(original, contours):
    """original - это исходная картинка в оттенках серого"""
    temp = original.copy()
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(temp, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return temp


# Получение контуров
def get_countours(image, original):
    # Инверсия цвета
    image = cv2.bitwise_not(image)

    # Формирование контуров
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contrs = [cv2.boundingRect(x) for x in contours]
    print('contrs before:', contrs)

    # Вывод исходного изображения и рядом все его контуры
    # all_contrs = plt_image_with_all_contours(original, contours)

    # Слияние близко расположенных контуров
    contrs = contour_processing(contrs)
    print('after: ', contrs)

    return contrs
