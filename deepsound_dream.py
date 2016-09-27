from sound_functions import get_fft
from sound_functions import save_wav
import numpy as np
from pprint import pprint

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

song = '../songsinmyhead/08dreams.wav'
label = 8
seconds = 4
frames_per_second = 5
model_def = 'soundnet/deepsound_simplenet3_8_deploy.prototxt'
model_weights = 'soundnet/simplenet3_8_5000.caffemodel'
solver_file = 'soundnet/simplesolver.prototxt'
restore_file = 'soundnet/auto_simple.solverstate'

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

	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]

	src.data[0][:] = mydata



	ox, oy = np.random.randint(-jitter, jitter+1, 2)
	ox = 0
	#src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
	

	net.forward(end=end)




	objective(dst)  # specify the optimization objective 

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



	# apply normalized ascent step to the input image
	ascent = step_size/np.abs(g).mean() * g

	print "ascent: "
	pprint(ascent)

	

	otherdata = np.add(mydata, ascent)
	#otherdata = np.roll(np.roll(otherdata, -ox, -1), -oy, -2) # unshift image

	return otherdata.copy()
	

def dream():
	caffe.set_mode_cpu()


	#model_weights = 'soundnet/simplenet3_5_500.caffemodel'
	#input_dim: 16
	#input_dim: 22050

	net = caffe.Net(model_def, model_weights, caffe.TRAIN)     

	[a_song_data, a_song_labels] = get_fft(song, seconds, label, 10000, frames_per_second=frames_per_second)



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

dream()


def dreamandlearn():


	input_data = np.array(sd, dtype=np.float32)
	input_labels = np.array(sl, dtype=np.float32)

	[input_data, input_labels, test_data, test_labels] = np.load('preprocessed_sound.npy')
	solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.SGDSolver(solver_file)
    solver.restore(restore_file)
    net = solver.net
    net.set_input_arrays(input_data, input_labels)
    testnet = solver.test_nets[0]
    testnet.set_input_arrays(test_data, test_labels)

    alldata = []

    step = make_step(net, input_data[0])

    for i in range(20):

    	alldata += [step]
    	step = make_step(net, step)
    	solver.step(1)

    save_wav("will_it_dream.wav", np.array(alldata))

