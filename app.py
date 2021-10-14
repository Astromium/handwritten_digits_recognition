from datetime import datetime
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime

model = load_model('digits_recognizer.h5')

def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # seperate the background elements from the foreground elements
    # ret, thresh = cv2.threshold(img.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     #create a rectangle around the digit
    #     cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness = 2) 
    #     # cropping the image (digit only)
    #     digit = thresh[y:y+h, x:x+w]
    #     #digit = cv2.resize(digit, (18, 18))
    #     # pad the image with 5 pixels in each side to reach 28x28
    #     digit = np.pad(digit, ((5,5), (5,5)), "constant", constant_values=0)
    
    digit = cv2.resize(img, (28, 28))
    #digit = (255-digit)
    # cv2.imshow('test', digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return digit

cap = cv2.VideoCapture(0)
w, h = 300, 300
cap.set(3, w)
cap.set(4, h)

interval = datetime.now().timestamp() * 1000
counter = 0

while cap.isOpened():
    t = datetime.now().timestamp() * 1000
    success, img = cap.read()
    cv2.imshow('capture', img)
    
    if t > interval + 5000:
        
        cv2.imwrite('rt.jpg', img)
        
        prp = preprocess_img('rt.jpg')
        cv2.imwrite('sample.jpg', prp)
        prp = prp.reshape((1, 28, 28, 1))
        prp = prp / 255.0
        preds = model.predict(prp)[0]
        print(np.argmax(preds), max(preds))
        os.remove('rt.jpg')
        interval = t
        counter += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    

# for i in range(9):
#     sample = preprocess_img("test" + str(i) + '.png')
#     sample = sample.reshape((1, 28, 28, 1))
#     sample = sample / 255.0

#     preds = model.predict(sample)[0]
#     print(i, np.argmax(preds), max(preds))

# print('images with smooth background')    

# for i in [0, 3, 4, 5, 6, 7, 8]:
#     sample = preprocess_img("valid" + str(i) + '.png')
#     sample = sample.reshape((1, 28, 28, 1))
#     sample = sample / 255.0

#     preds = model.predict(sample)[0]
#     print(i, np.argmax(preds), max(preds))    

# for i in range(9):
#     test = preprocess_img('test' + str(i) + '.png')
#     cv2.imwrite('test' + str(i) + '.png', test)
# test = preprocess_img('test2.png')
# cv2.imwrite('rip.jpg', test)

cap.release()
cv2.destroyAllWindows()


