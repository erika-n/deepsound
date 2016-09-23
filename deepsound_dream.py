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

def make_step(net, mydata, step_size=500, end='score',
	jitter=32, clip=True, objective=objective_L2, datanum=0):
	'''Basic gradient ascent step.'''

	#randomdata =  np.random.random_sample(mydata.shape)*3000 #kinda like jitter shift...
	
	#mydata[:] += randomdata
	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]
	middle = 'fc4'
	mid = net.blobs[middle]
	src.data[0][:] = mydata 
	#ox, oy = np.random.randint(-jitter, jitter+1, 2)
	#src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	
	#rc.data[0][:] = a_song_data[5][:]
	
	#net.forward(end=middle)
	net.forward(end=end)




	objective(dst)  # specify the optimization objective 
	dst.data[0][10:] = 1

	net.backward(start=end) 


	g = src.diff[0]
	# print "g: "
	# pprint(g)



	# print "dst: "
	# pprint(dst.data[0][:20])
	# print "dst diff: "
	# pprint(dst.diff[0][:20])
	# print "middle: "
	# pprint(mid.data[0][:20])
	# print "middle diff: "
	# pprint(mid.diff[0][:20])

	# sys.exit()

	# apply normalized ascent step to the input image
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

model_def = 'soundnet/deepsound_simplenet2_deploy.prototxt'
model_weights = 'soundnet/simplenet2.caffemodel'

net = caffe.Classifier(model_def, model_weights)     


[a_song_data, a_song_labels] = get_fft('../sounds/21truefaith.wav', 2, 18, 100)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

# print("data and label shapes")
sd = a_song_data[0:1]
sl = a_song_labels[0:1]

input_data = np.array(sd, dtype=np.float32)
input_labels = np.array(sl, dtype=np.float32)


#net.forward()

# print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
# print 'real class is: ' + str(input_labels[0])




alldata = [np.copy(input_data[0])]
#alldata += [np.copy(net.blobs['data'].data[0])]
step = make_step(net, np.copy(input_data[0]))
i = 0
for i in range(100):
	
	
	
	#net.blobs['data'].data[0][:] = step[:]
	if(i % 10== 0):

		print "step " + str(i)
		alldata += [step]
		#step = zoom(step)

	step = make_step(net, step)



save_wav("will_it_dream.wav", np.array(alldata))

