import cv2 as cv2
import cvlib as cv
import tensorflow.keras as tfk
from tensorflow.keras.models import load_model
import numpy as np 
import pandas as pd 
import face_recognition
from tkinter import *
from tkinter import filedialog,messagebox
from PIL import Image,ImageTk
from tensorflow.keras.preprocessing.image import img_to_array
import csv
from datetime import datetime


#File handing and Load Model

# # creating csv file
file =   open('Details.csv','w',newline='')
write = csv.writer(file)
write.writerow(['Age','Gender','Time']) #defining row
file.close() 
model = load_model('Senior.keras')#loading model


#creating functions for files iterating process and detecting gender age and picking colors based on criteria

def prediction(face_):
    val = model.predict(face_)# predicting face in the frame
    gender = "Female" if int(val[0][0][0]>0.5) else "Male"
    age  = round(val[1][0][0])
    color = (0,0,255) if age >=60 else (255,255,0)
    return age,gender,color


def write_file(age,gender):
    with open('Details.csv','a',newline='') as file:
        write = csv.writer(file)
        time = datetime.now().strftime('%H:%M:%S')
        write.writerow([age,gender,time])#writing file person details
        file.close()


#  Major Functionalities
#         1. Detecting Faces
#         2. Faces encoding process to avoid duplicate entries
#         3. Rectangles on faces and putting texts above 


previous = []# list for storing faces for face encoding process
age, gender = 0,0

win = Tk()# creating window
win.title('Detector')
win.geometry('600x400')
win.config(bg='gray')

# label for videos to display
label1 = Label(win,bg='gray')
label1.pack(side= 'right',padx=10,pady=10)

options = ['webcam',"video1.mp4","video2.mp4"] #options for testing custom videos

selected_option = StringVar()
selected_option.set("Test Videos")
files = StringVar() #to get selected file_path

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")]) # open file dialog choose file from computer
    if file_path:
           
            files.set(file_path)
            selected_option.set('Custom_videos')
    else:
        messagebox.showinfo("File Selection Cancelled", "No file selected.")# message shown if file is not selected


def get_selected_option():
   
    selected_options = selected_option.get() #default option
    
    if selected_options =='Test Videos' or selected_options=='Custom_videos':
        messagebox.showinfo("File Selection Cancelled", "No file selected.")

    elif selected_options=='webcam':
        files.set(selected_options)# sending video_path to read
        show_img()
    else:
        files.set(selected_options)
        show_img()

def show_img(): 
    if files.get():

        cap = cv2.VideoCapture(files.get())
           
        while cap.isOpened():
            status, frame = cap.read()                 # read frame from given input
            if status:
                
                frame = cv2.resize(frame, (400, 400))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                face, confidence = cv.detect_face(frame)      # apply face detection

                for idx, f in enumerate(face):                # loop through detected faces
                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
    
                    face_crop = frame[startY-10:endY+10,startX-10:endX+10]     # crop the detected face region
                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue
        
                    face_crop = cv2.resize(face_crop, (110,110))# resizing image for prediction
                    face_crop = face_crop.astype("float32") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)  # preprocessing for gender detection model
                    age,gender,color = prediction(face_crop)  # apply gender detection on face
            
                    cv2.rectangle(frame, (startX,startY), (endX,endY), color, 1)  # draw rectangle over face
                    label = f'age: {age} gender: {gender}' if age < 60 else f'Senior citizen gender: {gender}'
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7, (200, 255, 0), 2)  

                    # showing predicted age,gender
                try:            
                    img = Image.fromarray(frame) 
                    img_tk = ImageTk.PhotoImage(image=img)# converting image into tk format
            
                    label1.imgtk =img_tk
                    label1.configure(image= img_tk)# updating image inside label
                except:
                    win.after(50,show_img)# to avoid runtime error 
                    return
                win.update()
           
               
                face_encodings = face_recognition.face_encodings(frame, face) # face encoding to avoid duplicate data
                for encoding in face_encodings:
                    matches = face_recognition.compare_faces(previous, encoding) #comparing faces with previously stored
                    if not any(matches):
                        previous.append(encoding)# if not writing their details
                        write_file(age,gender) # sending details to csv file
            else:
                break
          
        cap.release()



#label for hold buttons below
label2 = Label(win,bg='gray')
label2.pack(side='left')

#menu for options to choose
dropdown = OptionMenu(label2, selected_option, *options)
dropdown.pack(padx=5,pady=5)

# Button for open file dialog box
select_video_btn = Button(label2, text="Choose File", command=select_video)
select_video_btn.pack(padx=5,pady=5)

# button to open custom videos
show = Button(label2,text='Show Selected',command=get_selected_option)
show.pack(padx=5,pady=5)

# Button to exit the application
exit_btn = Button(win, text="Exit", command=win.destroy,compound='bottom')
exit_btn.pack(side='bottom')


win.mainloop()






    