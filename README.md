# Vetegable Classification Using Transfer Learning

<center><img src= "https://p4.wallpaperbetter.com/wallpaper/667/254/333/vegetables-fruit-still-life-food-wallpaper-preview.jpg" alt ="Titanic" style='width:500px;'></center><br>

##Idea behind using Transfer learning
 - Instead of training a deep network from scratch, we can actually take a pre-trained network and use it for a different task as learning of a new task relies on the previously learned tasks.
 - Assisting in image analysis and classification tasks including object detection with good accuracy.
- We will use InceptionV3 which is a network already trained on more than a million images from the ImageNet database. Know about [InceptionV3](https://paperswithcode.com/method/inception-v3).

## Dataset Preparation
 We prepared the dataset Vegetable_Images included: <br />
 15 vegetable species, each species included:
| test  | train | validation |
| ----- | ----- | ---------- |
| 200  | 1000 | 200 |

You can overview dataset by using Dataset_overview.ipynb<br/>
- Couning the number of images in a folder. [test set](https://github.com/HaiDangAI/Vegetable_Classification/tree/main/Vegetable_Images/test) is our case.<br/>
![image](https://user-images.githubusercontent.com/85833803/168939132-96685fe4-5e58-4e86-9c65-223c15ea6524.png)<br/>
- Data Visualization (EDA)
![image](https://user-images.githubusercontent.com/85833803/168939689-b394a7ed-d410-4306-b2e4-a4cfb0d08353.png)

## Training Model
Use the command prompt run:
```
python TrainingModel.py
```
We take the data in [train folder](https://github.com/HaiDangAI/Vegetable_Classification/tree/main/Vegetable_Images/train) to train the model.
Training model process will take a lot of time. So that we already trained the model and saved it to **model_inceptionV3_epoch5.h5**
You can pass the training step and start to predict

## Predict and Evaluate
Use the command prompt run:
```
python Prediction.py
```
We have 3 options:
>1. Predict single image
>2. Predict single vegetable folder
>3. Predict test folder

1. Predict single image
Will require name of the file want to predict and enter to predict default image (Vegetable_Images\\test\\Bean\\0001.jpg)<br/>
Example filename: ```Vegetable_Images\test\Brinjal\0871.jpg```
The predict will plot the file image and name the model predicted.<br/>
![image](https://user-images.githubusercontent.com/85833803/170502826-69717fe1-9f22-4ff9-be5a-047195a1151f.png)

2. Predict single vegetable folder
Will require name of the folder want to predict and enter to predict default folder (Vegetable_Images\\test\\Bean)<br/>
![image](https://user-images.githubusercontent.com/85833803/170503107-9b1db324-7b80-488e-8085-49876910384c.png)

3. Predict test folder
Predict all file in test folder.<br/>
![image](https://user-images.githubusercontent.com/85833803/170505058-69047f8a-7de3-4690-bfb8-c916af3373bd.png)

## Graphical User Interface
```
python GUI.py
```
- Graphical User Interface when you run GUI.py<br/><br/>
<img src="https://user-images.githubusercontent.com/85833803/170506006-bf8ab36f-7056-441d-84c1-ffad2b824ca1.png" width="800" height="600">
- After upload the image.<br/><br/>
<img src="https://user-images.githubusercontent.com/85833803/170506081-9c248a55-3669-4474-8601-9ec421c8d451.png" width="800" height="600">
- Program will predict the name of vegetable and provide sone imformation about nutrition of the vegetable.<br/><br/>
<img src="https://user-images.githubusercontent.com/85833803/170506140-cd6309e9-f98f-45e0-850f-32691f93c8e5.png" width="800" height="600">


The Vegetable Classification program will help alot of thing about name and nutrition of vegetable.

    @inproceedings{Vetegable Classification,
    title     = {{Vetegable Classification Using Transfer Learning}},
    author    = {Hai Dang Nguyen, Ngo Sach Trung},
    year      = {2022}
    link      = {https://github.com/HaiDangAI/Vegetable_Classification}
    }
    
When you use the model **please cite** our link github.






