#!/usr/bin/env python

# Note: this script needs to be present at
#   /home/<your_user_name>/plantvillage/create_distribution.py
# and executed from within the
#   /home/<your_user_name>/plantvillage/
# directory

import glob
import os
import random
import shutil

TRAIN_PERCENTAGE = 70

TRAIN_SET = []
VAL_SET = []

# Distribute the files into Training and Validation sets
for _image in glob.glob("crowdai/*/*"):
    className = _image.split("/")[-2]

    # Some fileNames contain spaces, which creates some incompatibility with a preprocessing script shipped with caffe
    # Hence we replace all spaces in the filename with _
    newFileName = _image.split("/")[-1]
    newFileName = newFileName.replace(" ", "_")
    newFilePath = "crowdai/" + className + "/" + newFileName
    shutil.move(_image, newFilePath)

    if random.randint(0, 100) < TRAIN_PERCENTAGE:
        TRAIN_SET.append((newFilePath, className.split("_")[-1]))
    else:
        VAL_SET.append((newFilePath, className.split("_")[-1]))

# Write the distribution into a separate text files
try:
    os.mkdir("lmdb")
except:
    pass

f = open("lmdb/train.txt", "w")
for _entry in TRAIN_SET:
    f.write(_entry[0] + " " + _entry[1] + "\n")
f.close()

f = open("lmdb/val.txt", "w")
for _entry in VAL_SET:
    f.write(_entry[0] + " " + _entry[1] + "\n")
f.close()