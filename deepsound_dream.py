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

def make_step(net, mydata, step_size=10000, end='fc2',
	jitter=32, clip=True, objective=objective_L2, datanum=0):
	'''Basic gradient ascent step.'''

	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]
	src.data[0][:] = mydata 
	#ox, oy = np.random.randint(-jitter, jitter+1, 2)
	#src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	
	#rc.data[0][:] = a_song_data[5][:]
	
	print "src.data = "
	pprint(src.data)
	net.forward(end=end)
	objective(dst)  # specify the optimization objective 
	net.backward(start=end) 
	print "g ="
	g = src.diff[0]


	pprint(g)
	# apply normalized ascent step to the input imag
	ascent = step_size/np.abs(g).mean() * g
	print "mydata before: "
	pprint(mydata)
	print "ascent = "
	pprint (ascent)

	print mydata.shape
	print ascent.shape
	#fade = 0.3
	#otherdata1 = fade*mydata
	#fadeddata = mydata*fade
	#print "fadeddata: "
	#pprint(fadeddata)
	otherdata = np.add(mydata, ascent)
	#otherdata = np.roll(np.roll(otherdata, -ox, -1), -oy, -2) # unshift image
	print "otherdata: "
	pprint(otherdata)
	

	return np.copy(otherdata)
    #if clip:
    #    bias = net.transformer.mean['data']
    #    src.data[:] = np.clip(src.data, -bias, 255-bias)  


caffe.set_mode_cpu()

model_def = 'soundnet/deepsound_works_train.prototxt'
model_weights = 'soundnet/definitely.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


[a_song_data, a_song_labels] = get_fft('../sounds/19lovebites.wav', 0.5, 19, 100)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

print("data and label shapes")
sd = a_song_data[10:11]
sl = a_song_labels[10:11]

input_data = np.array(sd, dtype=np.float32)
input_labels = np.array(sl, dtype=np.float32)


net.set_input_arrays(input_data,input_labels )

net.forward()

print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
print 'real class is: ' + str(net.blobs['label'].data)



alldata = [np.copy(net.blobs['data'].data[0])]
#alldata += [np.copy(net.blobs['data'].data[0])]
step = make_step(net, net.blobs['data'].data[0])
i = 0
for i in range(50):
	print "step " + str(i)
	step = np.copy(make_step(net, step))
	#net.blobs['data'].data[0][:] = step[:]
	
	alldata += [step]





save_wav("will_it_dream.wav", np.array(alldata))

