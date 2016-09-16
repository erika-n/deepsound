from sound_functions import get_fft
from sound_functions import save_wav
import numpy as np

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

model_def = 'deepsound_production.prototxt'
model_weights = 'soundnet/soundnet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


[a_song_data, a_song_labels] = get_fft('../sounds/08wishlonger.wav', 2, 5)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

print("data and label shapes")


input_data = np.array(a_song_data, dtype=np.float32)
input_labels = np.array(a_song_labels, dtype=np.float32)
print input_data.shape
print input_labels.shape

net.set_input_arrays(input_data,input_labels )

net.forward()



print 'predicted class is:' + str(net.blobs['score'].data.argmax(1))
print net.blobs['score'].data
print net.blobs['loss'].data


net.backward()


outdata = net.blobs['conv2'].data

print outdata.shape

save_wav("will_it_dream.wav", outdata[0][0])

