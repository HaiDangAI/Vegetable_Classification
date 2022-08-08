import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


category={
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3 : 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13 : "Radish", 14: "Tomato"
}

def predict_image(filename,model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    
    plt.title("Prediction - {}".format(category[index]))
    plt.imshow(img_array)
    plt.show()

def predict_dir(filedir,model):
    pos=0
    images=[]
    total_images=len(os.listdir(filedir))
    true=filedir.split('\\')[-1]
    
    for i in sorted(os.listdir(filedir)):
        images.append(os.path.join(filedir,i))
        
    for subplot, imggg in enumerate(images):
        img_ = image.load_img(imggg, target_size=(224, 224))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0) 
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)
        
        pred=category.get(index)
        if pred==true:
            pos+=1

    acc=pos/total_images
    print("Accuracy for {orignal}: {:.2f} ({pos}/{total})".format(acc,pos=pos,total=total_images,orignal=true))

path_to_model='model_inceptionV3_epoch5.h5'
print("Loading the model..")
model = load_model(path_to_model)
print("Done!")

option = int(input("1. Predict single image\n2. Predict single vegetable folder\n3. Predict test folder\n"))

if option in [1,2,3]:
    if option == 1:
        filename = 'Vegetable_Images\\test\\Bean\\0001.jpg'
        filename = input('Enter filename: ') or filename
        predict_image(filename,model)
    elif option == 2:
        foldername = input('Enter folder name: ') or 'Vegetable_Images\\test\\Bean'
        predict_dir(foldername, model)
    else:
        for i in os.listdir('Vegetable_Images\\test'):
            predict_dir(os.path.join('Vegetable_Images\\test',i),model)
else:
    quit()
