import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)


path_to_model='model_inceptionV3_epoch5.h5'
print("Loading the model..")
model = load_model(path_to_model)
print("Done!")

category={
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3 : 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13 : "Radish", 14: "Tomato"
}

# def predict_image(filename,model):
#     img_ = image.load_img(filename, target_size=(224, 224))
#     img_array = image.img_to_array(img_)
#     img_processed = np.expand_dims(img_array, axis=0) 
#     img_processed /= 255.   
    
#     prediction = model.predict(img_processed)
#     index = np.argmax(prediction)
    
#     plt.title("Prediction - {}".format(category[index]))
#     plt.imshow(img_array)

def predict_image(img,model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    # im_np = np.asarray(im_pil)
    img_processed = np.expand_dims(im_pil, axis=0) 
    img_processed /= 255.
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    print(category[index])

while True:
    success, image = cap.read()
    predict_image(image,model)
    # print(category[index])
    cv2.imshow("Counting number of fingers", image)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break