import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
#load the trained model to classify sign
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

df = pd.read_csv('VegetableInformation.csv')
df = df.transpose()

category={
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3 : 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13 : "Radish", 14: "Tomato"
}

# load model
path_to_model='model_inceptionV3_epoch5.h5'
model = load_model(path_to_model)

# create GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

# predict image
def predict_image(filename):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    label.configure(foreground='#011638', text='Predicted - '+category[index])
    
    info.delete(1.0, END)
    info.insert(INSERT, df[index][0]+'\n'+df[index][1])
    info.pack(pady=10)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: predict_image(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
info = Text(top, width=40, height=20)
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()

top.mainloop()