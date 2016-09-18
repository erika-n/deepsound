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

def make_step(net, step_size=10000, end='fc1',
	jitter=32, clip=True, objective=objective_L2, datanum=0):
	'''Basic gradient ascent step.'''

	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]

	ox, oy = np.random.randint(-jitter, jitter+1, 2)
	src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

	net.forward(end=end)
	objective(dst)  # specify the optimization objective 
	net.backward(start=end) 
	g = src.diff[0]

	print g.shape
	print "g = "
	pprint(g)
	# apply normalized ascent step to the input imag
	ascent = step_size/np.abs(g).mean() * g

	fade = 0.2
	src.data[0,0,0:200, :] *= fade
	src.data[0,0,0:200, :] += ascent[0,0:200,:] 
	print "max ascent: "
	print np.max(ascent[0,0:200,:] )
	#src.data[0] = 0.75*src.data[0]/np.max(np.abs(src.data[0])) #normalize volume

	#src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image


	returning = np.copy(src.data[0])
	
	return returning
    #if clip:
    #    bias = net.transformer.mean['data']
    #    src.data[:] = np.clip(src.data, -bias, 255-bias)  


caffe.set_mode_cpu()

model_def = 'soundnet/deepsound_simplenet_train.prototxt'
model_weights = 'soundnet/notquite.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


[a_song_data, a_song_labels] = get_fft('../sounds/03orangecrush.wav', 0.5, 3, 1)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

print("data and label shapes")
sd = a_song_data
sl = a_song_labels

input_data = np.array(sd, dtype=np.float32)
input_labels = np.array(sl, dtype=np.float32)
print input_data.shape
print input_labels.shape

net.set_input_arrays(input_data,input_labels )

net.forward()

print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
print 'real class is: ' + str(net.blobs['label'].data)



alldata = []

i = 0
for i in range(10):
	print "step " + str(i)
	step = make_step(net, datanum=i)
	net.blobs['data'].data[0] = step
	alldata += [np.copy(step)]





save_wav("will_it_dream.wav", np.array(alldata))

