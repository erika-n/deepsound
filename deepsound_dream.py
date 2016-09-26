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
	dst.data[:] = 0

def zoom(mydata):

	newdata = np.ndarray((mydata.shape))
	for i in range(mydata.shape[1]/2):
		newdata[0,2*i, :] = mydata[0, i, :]
		newdata[0,2*i + 1, :] = mydata[0, i, :]

	return newdata

def make_step(net, mydata, step_size=100000, end='fc1',
	jitter=4, clip=True, objective=objective_L2):
	'''Basic gradient ascent step.'''

	#randomdata =  np.random.random_sample(mydata.shape)*3000 #kinda like jitter shift...
	
	#mydata[:] += randomdata
	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]
	# middle = 'fc2'
	# mid = net.blobs[middle]
	src.data[0][:] = mydata

	# print "src.data:"
	# pprint(src.data)

	ox, oy = np.random.randint(-jitter, jitter+1, 2)
	ox = 0
	#src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	
	#rc.data[0][:] = a_song_data[5][:]
	
	#net.forward(end=middle)
	net.forward(end=end)




	objective(dst)  # specify the optimization objective 
	#objective(mid)
	#dst.diff[0][19] = 1
	# enddiff = net.blobs[end].diff
	# myargs = {end: enddiff}
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

	print "ascent: "
	pprint(ascent)

	

	otherdata = np.add(mydata, ascent)
	#otherdata = np.roll(np.roll(otherdata, -ox, -1), -oy, -2) # unshift image
	#otherdata[:] -= randomdata
	
	return otherdata.copy()
	
    #if clip:
    #    bias = net.transformer.mean['data']
    #    src.data[:] = np.clip(src.data, -bias, 255-bias)  


caffe.set_mode_cpu()

model_def = 'soundnet/deepsound_simplenet3_8_deploy.prototxt'
model_weights = 'soundnet/simplenet3_8_5000.caffemodel'
#model_weights = 'soundnet/simplenet3_5_500.caffemodel'
#input_dim: 16
#input_dim: 22050

net = caffe.Net(model_def, model_weights, caffe.TRAIN)     


[a_song_data, a_song_labels] = get_fft('../songsinmyhead/08dreams.wav', 4, 8, 10000, frames_per_second=5)



#net.blobs['data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          120, 1470)  # image size is 227x227

# print("data and label shapes")
sd = a_song_data[1:2]
sl = a_song_labels[1:2]

input_data = np.array(sd, dtype=np.float32)
input_labels = np.array(sl, dtype=np.float32)
print input_data.shape
print net.blobs['data'].data.shape


net.blobs['data'].data[0][:] = input_data
#net.blobs['label'].data[0] = input_labels
net.forward()

print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
print 'real class is: ' + str(input_labels[0])




alldata = [np.copy(input_data[0])]
#alldata += [np.copy(net.blobs['data'].data[0])]
step = make_step(net, input_data[0])
i = 0
for i in range(20):
	
	
	
	#net.blobs['data'].data[0][:] = step[:]
	if(i % 1== 0):

		print "step " + str(i)
		alldata += [step]
		#step = zoom(step)

	step = make_step(net, step)



save_wav("will_it_dream.wav", np.array(alldata))

