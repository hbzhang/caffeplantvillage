import numpy as np
import matplotlib.pyplot as plts
from PIL import Image
import os
import glob
import logging




"""
# You can use the commented block of code below to
#  make sure that caffe is on the python path:
# This takes the path to caffe_root from the environment variable, so make sure
# the $CAFFE_ROOT environment variable is set
#
#
caffe_root = os.environ['CAFFE_ROOT']
import sys
sys.path.insert(0, caffe_root + 'python')
"""

import caffe


import sys


"""

logging

"""

file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('LOGGER_NAME')

"""
# Adapted from from : http://www.cc.gatech.edu/~zk15/deep_learning/classify_test.py
"""

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'deploy.prototxt'
PRETRAINED = 'snapshots/plantvillage.caffemodel' #plantvillage.caffemodel'
BINARY_PROTO_MEAN_FILE = "lmdb/mean.binaryproto"

"""
# Replicated from https://github.com/BVLC/caffe/issues/290
"""
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(BINARY_PROTO_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
mean = np.array(caffe.io.blobproto_to_array(blob))[0]

m_min, m_max = mean.min(), mean.max()
normal_mean = (mean - m_min) / (m_max - m_min)
in_shape=(227, 227)
mean = caffe.io.resize_image(normal_mean.transpose((1,2,0)),in_shape).transpose((2,0,1)) * (m_max - m_min) + m_min


##NOTE : If you do not have a GPU, you can uncomment the `set_mode_cpu()` call
#        instead of the `set_mode_gpu` call.
caffe.set_mode_gpu()
# caffe.set_mode_cpu()


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
     #   mean=mean_arr.mean(1).mean(1),
     mean = mean, 
     channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))

f = open("predict.csv", "w")
f.write(
    "filename,c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16,c_17,c_18,c_19,c_20,c_21,c_22,c_23,c_24,c_25,c_26,c_27,c_28,c_29,c_30,c_31,c_32,c_33,c_34,c_35,c_36,c_37\n")

number_of_files_processed = 0
for _file in glob.glob("test/*"):
    number_of_files_processed += 1
    FileName = _file.split("/")[-1]
    print _file
    input_image = caffe.io.load_image(_file)
    logging.info(FileName)
    #print input_image
    prediction = net.predict([input_image])
    s = FileName + ","
    for probability in prediction[0]:
        s += str(probability) + ","
    s = s[:-1] + "\n"
    f.write(s)
    print "Number of files : ", number_of_files_processed
    print 'predicted class:', prediction[0].argmax()
    print "**********************************************"
