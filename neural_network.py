import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from functions import *


# Инициализируем данные и метки
print("[INFO] Инициализируем данные и метки...")
trainX, trainY = prepare_data_and_labels(WIDTH)
testX, testY = prepare_data_and_labels(WIDTH, True)

# разбиваем данные на обучающую и тестовую выборки
# (trainX, testX, trainY, testY) = train_test_split(
#     data, labels, test_size=0.2, random_state=RAND_SEED)

print('trainX.shape=', trainX.shape)
print('testX.shape=', testX.shape)
print('trainY.shape=', trainY.shape)
print('testY.shape=', testY.shape)

# Трансформируем из двухмерного массива в трех мерный
trainX = trainX.reshape(-1, WIDTH, WIDTH, 1)
testX = testX.reshape(-1, WIDTH, WIDTH, 1)

# конвертируем метки из целых чисел в векторы
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# Определим функцию для выполнения между эпохами
class CaptchaResolveCallback(tf.keras.callbacks.Callback):
    def __init__(self, labelbinarizer):
        self.lb = labelbinarizer

    def prepare_captchas_and_labels(self):
        # Убрать вывод ненужной информации
        import os
        import sys
        f = open(os.devnull, 'w')
        stdout_obj = sys.stdout
        sys.stdout = f

        data = []
        labels = []

        _, captchas = split_captcha_for_testing()
        for captcha in captchas:
            data.extend(get_letter_images_from_image(captcha))
            labels.extend(captcha.split('/')[-1][:-4])

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        sys.stdout = stdout_obj
        return data, labels

    def on_epoch_end(self, epoch, logs=None):
        testX, labels = self.prepare_captchas_and_labels()
        testX = testX.reshape(-1, WIDTH, WIDTH, 1)
        predictions = self.model.predict(testX, batch_size=32)
        out_labels = self.lb.inverse_transform(predictions)
        count = 0
        errors_count = 0
        for i in range(0, len(out_labels), 5):
            errors = 0
            for j in range(5):
                if out_labels[i + j] != labels[i + j]:
                    errors += 1
            errors_count += errors
            if errors == 0:
                count += 1
        correct_count = len(out_labels) - errors_count
        print(f'Правильно распознано {count} из {CAPTCHA_COUNT}. '
              f'Кол-во правильных символов {correct_count} из {len(out_labels)}')
        self.model.captcha_result = getattr(self.model, 'captcha_result', [])
        self.model.captcha_result.append((epoch, count, correct_count, len(out_labels)))

# определим архитектуру с помощью Keras
model = Sequential()
# Добавляем слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(WIDTH, WIDTH, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Второй сверточный слой
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Третий сверточный слой
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Создаем вектор для полносвязной сети.
model.add(Flatten())
# Создадим однослойный перцептрон
# model.add(Dense(500, activation='sigmoid'))
# Создадим однослойный перцептрон
model.add(Dense(500, activation='sigmoid'))
# Создадим однослойный перцептрон
model.add(Dense(19, activation='softmax'))
model.summary()
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results = model.fit(
    trainX, trainY, epochs=EPOCHS, batch_size=32,
    validation_data=(testX, testY),
    callbacks=[CaptchaResolveCallback(lb)]
)

print_results_of_neural_network(model, results)
model.save('output/my_model.h5')  # Создание HDF5 файла 'my_model.h5'

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
draw_loss_figures(results, EPOCHS)
