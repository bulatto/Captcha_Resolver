# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    return dilation2


# Инверсия цвета
# cv2.bitwise_not(image)

# Нарисовать все контуры по очереди
# for i in range(len(contours)):
#     my = original.copy()
#     cv2.drawContours(my, contours, i, (255, 0, 0), 1)
#     plt_image('my', my)


# Посчитать кол-во вертикальных символов и вывести график
def calculate_vertical(img, show_plot=False):
    pix =[]
    for i in range(img.shape[1]):
        pix.append(sum([1 for j in range(img.shape[0]) if img[j][i] < 100]))
    if show_plot:
        plt.bar(list(range(img.shape[1])), pix)


# Найти минимальное кол-во вертикальных символов в окрестности x
def calculate_vertical(img, x):
    pix = []
    nx = 5
    r = range(min(0, x - nx), min(img.shape[1], x + nx))
    for i in r:
        pix.append((r, sum([1 for j in range(img.shape[0]) if img[j][i] < 100])))
    min_pix = min(pix)
    return [i for i, s in pix if s == min_pix][0]


# Обработка контуров
def contour_processing(contrs, image):

    def split_in_contour(i, j):
        ix, iy, iw, ih = contrs[i]
        jx, jy, jw, jh = contrs[j]
        # Разделение контура на несколько
        if (iw * ih - jw * jh) / 400 >= 1:
            print(f'Контур (jx={jx},jy={jy}) находится внутри (ix={ix},iy={iy})')
            lx, ly, lw, lh = ix, iy, jx - ix, ih
            if lw * lh > 400 and lw >= 13:
                print(f'Добавляем слева от (jx={jx},jy={jy}) контур (lx={lx},ly={ly})')
                contrs.append((lx, ly, lw, lh))
            rx, ry, rw, rh = jx + jw, iy, ix + iw - (jx + jw), ih
            if rw * rh > 400 and rw >= 13:
                print(f'Добавляем справа от (jx={jx},jy={jy}) контур (rx={rx},ry={ry})')
                contrs.append((rx, ry, rw, rh))
            print(f'Удаляем контур (ix={ix},iy={iy})')
            del contrs[i]

    # Удаление меньшего контура, если он полностью внутри другого
    def delete_contours_which_into_other(i, j):
        if contrs[i][2] * contrs[i][3] < contrs[j][2] * contrs[j][3]:
            i, j = j, i

        ix, iy, iw, ih = contrs[i]
        jx, jy, jw, jh = contrs[j]

        # Полностью внутри
        if iw * ih >= jw * jh:
            cases = [ix <= jx <= ix + iw, ix <= jx + jw <= ix + iw,
                     iy <= jy <= iy + ih, iy <= jy <= iy + ih]
            if all(cases):
                if jw * jh < 400:
                    del contrs[j]
                    print(f'Удаление контура (jx={jx},jy={jy}) так как он весь в (ix={ix},iy={iy})')
                else:
                    split_in_contour(i, j)
                return True

        # # Почти весь внутри
        if j == i + 1:
            w = min(ix + iw, jx + jw) - jx
            h = min(iy + ih, jy + jh) - max(iy, jy)
            proc = w*h / (jw * jh)
            if 0.8 <= proc <= 1.0:
                print(f'Контур (jx={jx},jy={jy}) на {round(proc * 100,2)}% внутри контура (ix={ix},iy={iy})')
                split_in_contour(i, j)
                return True

    # Соединение
    def joining(i, j):
        ix, iy, iw, ih = contrs[i]
        jx, jy, jw, jh = contrs[j]

        # Если внизу полноценная буква и наверху символ
        join_top = False
        if abs(ix - jx) < 10:
            if iy >= jy + jh:
                if iw * ih > 500:
                    del contrs[j]
                    print(f'Удаление контура (jx={jx},jy={jy}), он выше (ix={ix},iy={iy})')
                    return True
                else:
                    join_top = True
            elif jy >= iy + ih:
                if jw * jh > 500:
                    del contrs[i]
                    print(f'Удаление контура (ix={ix},iy={iy}), он выше (jx={jx},jy={jy})')
                    return True
                else:
                    join_top = True

        # Проверка что контуры имеют минимум 50% общего
        is_in_another = False
        if j == i + 1:
            area = (jx - ix) * ih  # Слева
            if iy < jy:
                area += (ix + iw - jx) * (jy - iy)  # Сверху
            if iy + ih > jy + jh:
                area += (ix + iw - jx) * (iy + ih - (jy + jh))  # Снизу
            common = 1 - area / (iw * ih)
            is_in_another = True if common >= 0.5 else False
            if is_in_another:
                print(f'Контур (ix={ix},iy={iy}) содержит более {round(common * 100,2)}% общего с (jx={jx},jy={jy})')

        # Если контуры маленькие и близкие
        if (abs(ix - jx) < 10) and (iw * ih + jw * jh < 600) or is_in_another or join_top:  # TODO: Было 15
            print(f'Соединение контуров (ix={ix},iy={iy}) и (jx={jx},jy={jy}) ...')
            x, y = min(ix, jx), min(iy, jy)
            w = max(ix + iw - x, jx + jw - x)
            h = max(iy + ih - y, jy + jh - y)
            if (w < 20 * 2 and h < 20 *2) or is_in_another or join_top:
                contrs[i] = (x, y, w, h)
                del contrs[j]
                return True
        return False

    # Разъединение контура
    def separate(i):
        ix, iy, iw, ih = contrs[i]
        k = iw // 20
        k = min(k, 5 + 1 - len(contrs))  # TODO: Добавлено, проверить
        if k > 1:
            print(f'Разъединение контура (ix={ix},iy={iy}) на {k} частей...')
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

    def run_func_to_all_contrs(func):
        length = len(contrs)
        for i in range(length):
            for j in range(i + 1, length):
                # Если один контур внутри другого
                change = func(i, j)
                if change:
                    return True
        return

    # Выполнение соединения и разъединения контуров
    def contours_changing(contrs):
        length = len(contrs)
        for i in range(length):
            for j in range(i + 1, length):
                # Если один контур внутри другого
                was_change = delete_contours_which_into_other(i, j)
                if was_change:
                    return True
                # Соединение
                was_change = joining(i, j)
                if was_change:
                    return True

        for i in range(length):
            for j in range(i + 1, length):
                # Разделение
                was_change = separate(i)
                if was_change:
                    return True
        return False

    # Удаление ненужных контуров
    len_before = len(contrs)
    contrs = [c for c in contrs if 50 <= c[2] * c[3]]
    print(f'Удалено {len_before - len(contrs)} контуров  площадью меньше 50')

    was_change = True
    while was_change:
        contrs = sorted(contrs)
        was_change = contours_changing(contrs)

    # Удаление ненужных контуров
    len_before = len(contrs)
    contrs = [c for c in contrs if 100 <= c[2] * c[3]]
    print(f'Удалено {len_before - len(contrs)} контуров  площадью меньше 100')

    # Если перебор букв
    while len(contrs) > 5:
        sizes = [c[2] * c[3] for c in contrs]
        min_size = min(sizes)
        ind = [i for i, c in enumerate(contrs) if c[2] * c[3] == min_size][0]
        del contrs[ind]

    # Если не хватает букв
    while len(contrs) < 5:
        w = [c[2] for c in contrs]
        w_max_ind = [i for i, c in enumerate(contrs) if c[2] == max(w)][0]

        dx = contrs[w_max_ind][2] // 2
        zx, zy, zw, zh = contrs[w_max_ind]
        print(f"Букв не хватает. Делим один контур пополам (ix={zx},iy={zy})")
        contrs[w_max_ind] = (zx, zy, dx, zh)
        contrs.append((zx + dx, zy, dx, zh))

    # Если последний элемент маленький то соединяем и делим
    if contrs[-1][2] * contrs[-1][3] < 300:
        print('Последний элемент маленький,соединяем и делим')
        ix, iy, iw, ih = contrs[-2]
        jx, jy, jw, jh = contrs[-1]
        x, y = min(ix, jx), min(iy, jy)
        w = max(ix + iw - x, jx + jw - x)
        h = max(iy + ih - y, jy + jh - y)
        contrs[-2] = (x, y, w, h)
        del contrs[-1]

        ix, iy, iw, ih = contrs[-1]
        dx = iw // 2
        contrs[-1] = (ix, iy, dx, ih)
        contrs.append((ix + dx, iy, dx, ih))
    return contrs


# Накладывание контуров на изображение
def contours_to_image(image, contrs):
    output = image.copy()
    for i in contrs:
        x, y, w, h = i
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return output


# Получение изображения со всеми контурами
def get_image_with_all_contours(original, contours):
    """original - это исходная картинка в оттенках серого"""
    temp = original.copy()
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(temp, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return temp


# Получение контуров
def get_countours(image, original, need_all_contrs=False):
    # Инверсия цвета
    image = cv2.bitwise_not(image)

    # Формирование контуров
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contrs = sorted([cv2.boundingRect(x) for x in contours])
    print('contrs before:', contrs)

    # Обработка контуров
    contrs = contour_processing(contrs, image)
    print('after: ', contrs)

    if need_all_contrs:
        # Получение изображения со всеми контурами
        all_contrs = get_image_with_all_contours(original, contours)
        return contrs, all_contrs
    else:
        return contrs


# строим графики потерь и точности
def draw_loss_figures(results, EPOCHS):
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, results.history["loss"], label="train_loss")
    plt.plot(N, results.history["val_loss"], label="val_loss")
    plt.plot(N, results.history["accuracy"], label="train_acc")
    plt.plot(N, results.history["val_accuracy"], label="val_acc")
    if results.history.get('correct captha') and results.history.get('correct letters'):
        plt.plot(N, results.history['correct captha'], label='corr_captha')
        plt.plot(N, results.history['correct letters'], label='corr_let')
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('output/Figure Accuracy')


def get_images_before_and_after_contouring(img_url):
    img_c1 = cv2.imread(img_url, cv2.CV_8UC1)
    img_otsu = image_to_otsu(img_c1)
    contours, img_all_contrs = get_countours(img_otsu, img_c1, True)
    # Накладывание контуров на изображение
    result = contours_to_image(img_c1, contours)
    # Сравнение изображений
    plt_sub(1, 3, [img_otsu, img_all_contrs, result], ['После обработки', 'Все контуры', 'Результат'])
