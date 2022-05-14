import os, pathlib
import pandas as pd

dataset='Vegetable_Images'

train_folder = os.path.join(dataset,"train")
test_folder = os.path.join(dataset,"validation")
validation_folder = os.path.join(dataset,"test")



# def count_files(rootdir):
#     '''counts the number of files in each subfolder in a directory'''
#     for path in pathlib.Path(rootdir).iterdir():
#         if path.is_dir():
#             print("There are " + str(len([name for name in os.listdir(path) \
#             if os.path.isfile(os.path.join(path, name))])) + " files in " + \
#             str(path.name))

def count_files(rootdir):
    CountTable = pd.DataFrame()
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            CountTable[str(path.name)] = [len([name for name in os.listdir(path)])]
    return CountTable

print('Test Folder')
print(count_files(os.path.join(test_folder)))
print('Train Folder')
print(count_files(os.path.join(train_folder)))
print('Validation Folder')
print(count_files(os.path.join(validation_folder)))


