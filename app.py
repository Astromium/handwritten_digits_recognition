import PIL
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageGrab
import numpy as np
from tkinter import *
import tkinter as tk
import win32gui
import cv2
import os
from datetime import datetime

model = load_model('digits_recognizer.h5')


def get_handle():
    toplist = []
    windows_list = []
    canvas = 0
    def enum_win(hwnd, result):
        win_text = win32gui.GetWindowText(hwnd)
        windows_list.append((hwnd, win_text))

    win32gui.EnumWindows(enum_win, toplist)
    for (hwnd, win_text) in windows_list:
        if 'tk' == win_text:
            canvas = hwnd

    return canvas            

def preprocess_img(path):
    
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #seperate the background elements from the foreground elements
        _, thresh = cv2.threshold(img.copy(), 75, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            #create a rectangle around the digit
            cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness = 2) 
            # cropping the image (digit only)
            digit = thresh[y:y+h, x:x+w]
            resized = cv2.resize(digit, (18, 18))
            # pad the image with 5 pixels in each side to reach 28x28
            digit = np.pad(resized, ((5,5), (5,5)), "constant", constant_values=0)
            preprocessed_digit = (digit)

        return preprocessed_digit

   

def predict_handwritten_digit(img):
    path = "predic.jpg"
    cv2.imwrite(path, img)
    sample = preprocess_img(path)
    sample = sample.reshape((1, 28, 28, 1))
    sample = sample / 255.0
    preds = model.predict(sample)[0]
    digit, acc = np.argmax(preds), max(preds)
    os.remove(path)

    return digit, acc

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 28))
        self.classify_btn = tk.Button(self, text = "Recognize", command=self.predict_digit)
        self.button_clear = tk.Button(self, text = "Clear", command=self.clear_all)
        # grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def predict_digit(self):
        HWND = self.canvas.winfo_id()
        hwnd = get_handle()
        rect = win32gui.GetWindowRect(HWND)
        x1, y1, x2, y2 = rect
        img = ImageGrab.grab((x1+40, y1+40, x2+100, y2+100))
        #print(type(img))
        # convert to ndarray
        img = np.asarray(img)
        cv2.imwrite("ape.jpg", img)
        #print(type(img))
        digit, acc =  predict_handwritten_digit(img)       
        self.label.configure(text = "prediction: " + str(digit) + "\nconfidence: " + str(int(acc*100)) + " %")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill="black") 

# test = preprocess_img('ape.jpg')
# cv2.imwrite('ape.jpg', test)

app = App()
mainloop()           