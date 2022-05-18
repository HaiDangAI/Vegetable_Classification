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


