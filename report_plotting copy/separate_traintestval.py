"""
This file is used to separate TLA4AE dataset into train:test:validation = 3:1:1 (same as SWET)

"""
import os
import random
import splitfolders
import shutil


# Change them into your own directories
TLA_DIR = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Original_datasets/TLA4AE_grouped_Feb_Apr2022'
lABEL_DIR = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Original_datasets/binary_masks_TLA'
OUTPUT_DIR = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_training_data/dataset_TLA'


# Function written by Zihao Wang, descriptions as below
def listdir_nohidden(path, is_train=True):
    """Generate a list contains all the non-hidden files in the given path,
    this is to exclude system hidden files such as .DS_Store
    - feature added: sort the list to prevent images and masks are not 1-to-1 corresponded.

    # Arguments
        path: directory to the image folder

        is_train: boolean variable shows whether this function is used in model training

    # Returns
        return a sorted list which contains all the non-hidden files in the given path

    """
    file_list = [file for file in os.listdir(path) if not file.startswith('.')]
    # print('before', file_list)
    if not is_train:
        if any("_aug" in string for string in file_list):
            file_list = [f.replace('_aug', '') for f in file_list]
            file_list = sorted(file_list)
            file_list = [f.replace('.jpg', '_aug.jpg') for f in file_list]
        # print("aug detected!")
        else:
            file_list = sorted(file_list)


    return file_list


def find_id_lists(TLA_list):

    for seed in range(100):

        random.seed(seed)
        random.shuffle(TLA_list)
        test_set_id = TLA_list[0:int(len(TLA_list)*0.2)]
        val_set_id = TLA_list[int(len(TLA_list)*0.2):int(len(TLA_list)*0.4)]
        train_set_id = TLA_list[int(len(TLA_list)*0.4):]


        """
            Counting numbers in each set
        """
        train_num = 0
        for ids in train_set_id:
            ID_DIR = os.path.join(TLA_DIR, ids)
            train_num += len(listdir_nohidden(ID_DIR))


        test_num = 0
        for ids in test_set_id:
            ID_DIR = os.path.join(TLA_DIR, ids)
            test_num += len(listdir_nohidden(ID_DIR))



        val_num = 0
        for ids in val_set_id:
            ID_DIR = os.path.join(TLA_DIR, ids)
            val_num += len(listdir_nohidden(ID_DIR))

        
        if test_num == val_num:
            # print(test_num)
            # print(val_num)
            # print(TLA_list)
            return test_set_id, val_set_id, train_set_id

    return [], [], []




TLA_list = sorted(listdir_nohidden(TLA_DIR))
# label_list = sorted(listdir_nohidden(LABEL_DIR))

# Should be 33
# print(len(TLA_list))
# print(label_list[1:5])


"""
    This is to find '.jfif' files (and remember to change them, because OpenCV does not read such file types)
"""
# for strings in TLA_list:
#     if '.jfif' == strings[-5:]:
#         print(strings)


"""
    Find the missing file in TLA_list directory
"""
# print(list(set(label_list).difference(TLA_list)))


"""
    Split the images into training, testing and validation sets based on patient IDs
    6:2:2
"""

test_set_id, val_set_id, train_set_id = find_id_lists(TLA_list)

if test_set_id and val_set_id and train_set_id:

    train_set_real = os.path.join(OUTPUT_DIR, "training_set_TLA/reals")
    test_set_real = os.path.join(OUTPUT_DIR, "test_set_TLA/reals")
    val_set_real = os.path.join(OUTPUT_DIR, "validation_set_TLA/reals")

    # create folder if not exist
    if not os.path.exists(train_set_real):
        os.makedirs(train_set_real)
        # os.system("mkdir -p " + train_set_real)
    if not os.path.exists(test_set_real):
        os.makedirs(test_set_real)
        # os.system("mkdir -p " + test_set_real)
    if not os.path.exists(val_set_real):
        os.makedirs(val_set_real)
        # os.system("mkdir -p " + val_set_real)

    # Put images in training set
    for ids in train_set_id:
        ID_DIR = os.path.join(TLA_DIR, ids)
        for imgs in listdir_nohidden(ID_DIR):
            shutil.copy(os.path.join(ID_DIR, imgs), train_set_real)

    # Put images in test set
    for ids in test_set_id:
        ID_DIR = os.path.join(TLA_DIR, ids)
        for imgs in listdir_nohidden(ID_DIR):
            shutil.copy(os.path.join(ID_DIR, imgs), test_set_real)


    # Put images in test set
    for ids in val_set_id:
        ID_DIR = os.path.join(TLA_DIR, ids)
        for imgs in listdir_nohidden(ID_DIR):
            shutil.copy(os.path.join(ID_DIR, imgs), val_set_real)

else:
    print('Split dataset failed')



        
"""
    Organise correpsonding masks according to real images
"""
train_set_labels = os.path.join(OUTPUT_DIR, "training_set_TLA/labels")
test_set_labels = os.path.join(OUTPUT_DIR, "test_set_TLA/labels")
val_set_labels = os.path.join(OUTPUT_DIR, "validation_set_TLA/labels")

# create folder if not exist
if not os.path.exists(train_set_labels):
    os.makedirs(train_set_labels)
    # os.system("mkdir -p " + train_set_real)
if not os.path.exists(test_set_labels):
    os.makedirs(test_set_labels)
    # os.system("mkdir -p " + test_set_real)
if not os.path.exists(val_set_labels):
    os.makedirs(val_set_labels)
    # os.system("mkdir -p " + val_set_real)

# Put images in training set
for ids in train_set_id:
    ID_DIR = os.path.join(TLA_DIR, ids)
    for imgs in listdir_nohidden(ID_DIR):
        shutil.copy(os.path.join(lABEL_DIR, imgs), train_set_labels)

# Put images in test set
for ids in test_set_id:
    ID_DIR = os.path.join(TLA_DIR, ids)
    for imgs in listdir_nohidden(ID_DIR):
        shutil.copy(os.path.join(lABEL_DIR, imgs), test_set_labels)


# Put images in test set
for ids in val_set_id:
    ID_DIR = os.path.join(TLA_DIR, ids)
    for imgs in listdir_nohidden(ID_DIR):
        shutil.copy(os.path.join(lABEL_DIR, imgs), val_set_labels)



"""
    Make sure the image and label outputs are the same
"""
train_dir_image = os.path.join(OUTPUT_DIR, "training_set_TLA/reals")
test_dir_image = os.path.join(OUTPUT_DIR, "test_set_TLA/reals")
val_dir_image = os.path.join(OUTPUT_DIR, "validation_set_TLA/reals")

train_dir_label = os.path.join(OUTPUT_DIR, "training_set_TLA/labels")
test_dir_label = os.path.join(OUTPUT_DIR, "test_set_TLA/labels")
val_dir_label = os.path.join(OUTPUT_DIR, "validation_set_TLA/labels")


train_list_image = sorted(listdir_nohidden(train_dir_image))
test_list_image = sorted(listdir_nohidden(test_dir_image))
val_list_image = sorted(listdir_nohidden(val_dir_image))

train_list_label = sorted(listdir_nohidden(train_dir_label))
test_list_label = sorted(listdir_nohidden(test_dir_label))
val_list_label = sorted(listdir_nohidden(val_dir_label))


if train_list_image == train_list_label:
    print('Train set OK!')
if test_list_image == test_list_label:
    print('Test set OK!')
if val_list_image == val_list_label:
    print('Val set OK!')








