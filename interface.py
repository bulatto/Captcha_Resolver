# coding: utf-8
from keras.models import load_model
import pickle
from os import path
from tkinter import *
from tkinter import filedialog, messagebox

from functions import *


FILE = None
with open('LabelBinarizer', 'rb') as f:
    lb = pickle.load(f)


def get_file():
    global FILE
    try:
        FILE = filedialog.askopenfilename(initialdir=path.dirname(__file__), filetypes=(("PNG images", "*.png"),))
        btnRecognize.pack(side=BOTTOM)
    except Exception as e:
        print(f'Произошла ошибка при выборе файла: {e}')
        messagebox.showinfo('Ошибка', 'Файл не выбран!')
        FILE = None
        btnRecognize.pack_forget()
        return


def recognize():
    if not FILE:
        messagebox.showinfo('Ошибка', 'Файл не выбран!')
        return
    cdata = np.array(get_letter_images_from_image(FILE), dtype="float") / 255.0
    cdata = cdata.reshape(-1, WIDTH, WIDTH, 1)
    model = load_model('model.h5')
    predictions = model.predict(cdata)
    out_labels = lb.inverse_transform(predictions)
    result = ''.join(out_labels)
    get_images_before_and_after_contouring(FILE, result)


window = Tk()
window.title(f"Распознавание капчи")
font = ("Arial", 13)

label = Label(window, text="Выбор изображения:", font=font)
label.place(relx=0, rely=0.15)

btnFile = Button(window, text="Обзор", font=font, command=get_file)
btnFile.place(relx=.7, rely=0.15)

btnRecognize = Button(window, text="Распознать", font=font, command=recognize)
btnRecognize.pack(side=BOTTOM)

window.geometry('400x100')
window.mainloop()
