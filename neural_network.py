import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from functions import *


# Инициализируем данные и метки
print("[INFO] Инициализируем данные и метки...")
data = []
labels = []
EPOCHS = 20

# Получаем случайный список изображений
image_paths = get_all_letters_list()

for image_path in image_paths:
    # загружаем изображение, меняем размер на 32x32 пикселей (без учёта
    # соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
    # добавляем в список
    image = cv2.imread(image_path)
    image = cv2.resize(image, (30, 30))  #  .flatten()
    data.append(image)

    # извлекаем метку класса из пути к изображению и обновляем
    # список меток
    label = image_path.split('/')[-2]
    labels.append(label)


# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=rand_seed)

print('trainX.shape ',trainX.shape)
print('testX.shape ',testX.shape)
print('trainY.shape ',trainY.shape)
print('testY.shape ',testY.shape)

# Трансформируем из двухмерного массива в трех мерный(28х28х1 канал)
# trainX = trainX.reshape(1194,30,30,1)
# testX = testX.reshape(398,30,30,1)

# конвертируем метки из целых чисел в векторы
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# определим архитектуру с помощью Keras
model = Sequential()
# Добавляем слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(30, 30, 3)))
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
model.add(Dense(20, activation='relu'))
# Создадим однослойный перцептрон
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results = model.fit(
    trainX, trainY,
    epochs=EPOCHS,
    batch_size=32,
    validation_data=(testX, testY)
)
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, results.history["loss"], label="train_loss")
plt.plot(N, results.history["val_loss"], label="val_loss")
plt.plot(N, results.history["accuracy"], label="train_acc")
plt.plot(N, results.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('figures/Figure Accuracy')