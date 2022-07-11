#!/usr/bin/python

# test file
# TODO:
# 	Figure out four point transform
#	Figure out testing data warping
# 	Use webcam as input
# 	Figure out how to use contours
# 		Currently detects inner rect -> detect outermost rectangle
# 	Try using video stream from android phone

import cv2
import utils as u 
from matplotlib import pyplot as plt
import numpy as np


from tkinter import *
from tkinter import filedialog
root =Tk()
root.title("Currency Detector")
root.geometry("500x500")
root.config(background="sky blue")
ll=Label(text="Jaypee University of Information Technology",fg="Green" ,bg ="sky blue",width="110" , height="5",font=('times',18,'bold'))
ll.place(x=00,y=00)

ll=Label(text="Welcome to Currency Detection System",fg="red" ,width="135" , height="5",font=('times',16,'bold'))
ll.place(x=00,y=100)


lm=Label(text="1.Choose the currency by clicking on Run Button",fg="red",width="135" ,height='2', font=('times',15,'bold'))
lm.place(x=00,y=200)
lv=Label(text="2.Press Enter To Identify the Currency",fg="red", width="135" ,font=('times',15,'bold'))
lv.place(x=00,y=235)
lvb=Label(text="3.If Currency Matches , the Picture of currency will be displayed", width="135" ,fg="red",font=('times',15,'bold'))
lvb.place(x=00,y=260)
ll=Label(text="Designed & Developed By:",fg="red" ,bg="sky blue" , height="5",font=('times',12,'bold'))
ll.place(x=00,y=630)
ll=Label(text="Ayush Gupta",fg="red" ,bg="sky blue" , width="40", font=('times',12,'bold'))
ll.place(x=00,y=700)

def abc():
    filename =filedialog.askopenfilename()
    print(filename)
    return filename

def xyz():
    max_val = 8
    max_pt = -1
    max_kp = 0
    
    orb = cv2.ORB_create()
    location = abc()
    test_img = u.read_img(location)
    original = u.resize_img(test_img, 0.4)
    u.display('original', original)
    (kp1, des1) = orb.detectAndCompute(test_img, None)
    
    training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']
    
    for i in range(0, len(training_set)):
        train_img = cv2.imread(training_set[i])
        (kp2, des2) = orb.detectAndCompute(train_img, None)
        bf = cv2.BFMatcher()    
        all_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for (m, n) in all_matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        
        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            max_kp = kp2
        print(i, ' ', training_set[i], ' ', len(good))
    
    if max_val != 8:
        print(training_set[max_pt])
        print('good matches ', max_val)
        
        train_img = cv2.imread(training_set[max_pt])
        img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
        note = str(training_set[max_pt])[6:-4]
        print('\nDetected denomination: Rs. ', note)
        (plt.imshow(img3), plt.show())
        u.display('feature', img3)
    else:
        print('No Matches')
    

btn1=Button(root,text="Choose Image",width=50,height=7, bg="yellow",command=xyz)
btn1.place(x=600,y=450)

root.mainloop()

