from sound_functions import get_fft
from sound_functions import save_wav
import numpy as np
from pprint import pprint

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe




def objective_L2(dst):
	dst.diff[:] = dst.data 

def make_step(net, step_size=500, end='conv1',
	jitter=32, clip=True, objective=objective_L2):
	'''Basic gradient ascent step.'''

	src = net.blobs['dummydata'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]

    # ox, oy = np.random.randint(-jitter, jitter+1, 2)
    # oy = 0
    # src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

	net.forward(end=end)
	objective(dst)  # specify the optimization objective
	net.backward(start=end)
	g = src.diff[0]
	print "g = "
	pprint(g)
	print "g max min: "
	print np.max(g)
	print np.min(g)
	print g.shape

	# apply normalized ascent step to the input imag
	src.data[:] += step_size/np.abs(g).mean() * g
	return src.data[0] 
    #src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    #if clip:
    #    bias = net.transformer.mean['data']
    #    src.data[:] = np.clip(src.data, -bias, 255-bias)  


caffe.set_mode_cpu()

model_def = 'deepsound_production.prototxt'
model_weights = 'soundnet/soundnet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TRAIN)     # use test mode (e.g., don't perform dropout)


[a_song_data, a_song_labels] = get_fft('../sounds/03orangecrush.wav', 1, 3)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

print("data and label shapes")


input_data = np.array(a_song_data[15:], dtype=np.float32)
input_labels = np.array(a_song_labels[15:], dtype=np.float32)
print input_data.shape
print input_labels.shape

net.set_input_arrays(input_data,input_labels )

net.forward()
net.forward()
net.forward()
net.forward()
print 'predicted class is:' + str(net.blobs['score'].data.argmax(1))
print net.blobs['score'].diff
print net.blobs['loss'].diff

alldata = []
alldata += [make_step(net)]
alldata += [make_step(net)]
alldata += [make_step(net)]
alldata += [make_step(net)]
alldata += [make_step(net)]





save_wav("will_it_dream.wav", np.array(alldata))

