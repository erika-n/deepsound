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

def zoom(mydata):

	newdata = np.ndarray((mydata.shape))
	for i in range(mydata.shape[1]/2):
		newdata[0,2*i, :] = mydata[0, i, :]
		newdata[0,2*i + 1, :] = mydata[0, i, :]

	return newdata

def make_step(net, mydata, step_size=500, end='fc4',
	jitter=32, clip=True, objective=objective_L2, datanum=0):
	'''Basic gradient ascent step.'''

	#randomdata =  np.random.random_sample(mydata.shape)*3000 #kinda like jitter shift...
	
	#mydata[:] += randomdata
	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]
	src.data[0][:] = mydata 
	#ox, oy = np.random.randint(-jitter, jitter+1, 2)
	#src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	
	#rc.data[0][:] = a_song_data[5][:]
	

	net.forward(end=end)
	objective(dst)  # specify the optimization objective 
	net.backward(start=end) 

	g = src.diff[0]


	# apply normalized ascent step to the input imag
	ascent = step_size/np.abs(g).mean() * g


	#fade = 0.3
	#otherdata1 = fade*mydata
	#fadeddata = mydata*fade
	#print "fadeddata: "
	#pprint(fadeddata)
	otherdata = np.add(mydata, ascent)
	#otherdata = np.roll(np.roll(otherdata, -ox, -1), -oy, -2) # unshift image
	#otherdata[:] -= randomdata
	
	return np.copy(otherdata)
	
    #if clip:
    #    bias = net.transformer.mean['data']
    #    src.data[:] = np.clip(src.data, -bias, 255-bias)  


caffe.set_mode_cpu()

model_def = 'soundnet/deepsound_simplenet2_train.prototxt'
model_weights = 'soundnet/simplenet2.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


[a_song_data, a_song_labels] = get_fft('../sounds/10brandenburg2.wav', 2, 10, 100)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

print("data and label shapes")
sd = a_song_data[50:51]
sl = a_song_labels[50:51]

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
for i in range(100):
	
	step = make_step(net, step)
	
	#net.blobs['data'].data[0][:] = step[:]
	if(i % 10== 0):

		print "step " + str(i)
		alldata += [step]
		#step = zoom(step)





save_wav("will_it_dream.wav", np.array(alldata))

