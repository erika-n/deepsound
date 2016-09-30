from sound_functions import get_fft
from sound_functions import save_wav
import numpy as np
from pprint import pprint
from shutil import copyfile

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

song = '../songsinmyhead/04coldhearted.wav'
label = 8
seconds = 1
frames_per_second = 60
model_def = 'soundnet/auto_small_deploy.prototxt'
model_weights = 'soundnet/small_iter_5000.caffemodel'
solver_file = 'soundnet/smallsolver.prototxt'
restore_file = 'soundnet/small_iter_500.solverstate'

def objective_L2(dst):
	dst.diff[:] = dst.data 
	

def zoom(mydata):

	newdata = np.ndarray((mydata.shape))
	for i in range(mydata.shape[1]/2):
		newdata[0,2*i, :] = mydata[0, i, :]
		newdata[0,2*i + 1, :] = mydata[0, i, :]

	return newdata

def make_step(net, mydata, step_size=0.5, end='fc1',
	jitter=4, clip=True, objective=objective_L2, label=None):
	'''Basic gradient ascent step.'''

	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]


	
	src.data[0][:] = mydata




	net.forward(end=end)

	

	objective(dst)  # specify the optimization objective 

	#dst.diff[0][:] = np.random.random_sample(dst.diff[0].shape)

	# if label:
	# 	dst.diff[0][label] = 1000

	net.backward(start=end) 


	g = src.diff[0]

	print "g: "
	pprint(g)

	# normalized ascent step 
	ascent = step_size/np.abs(g).mean() * g

	print "ascent: "
	pprint(ascent)

	

	otherdata = ascent 
	print "otherdata: "
	pprint (otherdata)
	#otherdata = np.roll(np.roll(otherdata, -ox, -1), -oy, -2) # unshift image

	return otherdata.copy()
	

def dream():
	print "dreaming..."
	caffe.set_mode_cpu()


	#model_weights = 'soundnet/simplenet3_5_500.caffemodel'
	#input_dim: 16
	#input_dim: 22050

	net = caffe.Net(model_def, model_weights, caffe.TRAIN)     

	[a_song_data, a_song_labels] = get_fft(song, seconds, label, 200, frames_per_second=frames_per_second)



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




	alldata = []
	#alldata += [np.copy(net.blobs['data'].data[0])]
	step = make_step(net,np.array(np.zeros(input_data[0].shape)))
	
	l = 0
	for i in range(10):
		l = (l + 1) %30
		
		
		#net.blobs['data'].data[0][:] = step[:]
		if(i % 1== 0):

			print "step " + str(i)
			alldata += [step]
			#step = zoom(step)


		step = make_step(net, step, label=l)



	save_wav("will_it_dream.wav", np.array(alldata))




def dreamandlearn():
	# print "copying files..."
	# copyfile( model_weights + ".orig", model_weights)
	# copyfile(restore_file + ".orig", restore_file)


	[input_data, input_labels, test_data, test_labels] = np.load('preprocessed_sound.npy')
	solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
	solver = caffe.SGDSolver(solver_file)
	solver.restore(restore_file)
	net = solver.net
	net.set_input_arrays(input_data, input_labels)
	testnet = solver.test_nets[0]
	testnet.set_input_arrays(test_data, test_labels)




	[a_song_data, a_song_labels] = get_fft(song, seconds, label, 10000, frames_per_second=frames_per_second)



	song_data = np.array(a_song_data[1:2], dtype=np.float32)

	net.blobs['data'].data[0][:] = song_data
	#net.blobs['label'].data[0] = input_labels
	net.forward()

	print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
	print 'real class is: ' + str(label)



	alldata = []

	step = make_step(net,np.zeros(song_data[0].shape))

	for i in range(30):
		print "step: " + str(i)
		# step = make_step(net, np.zeros(song_data[0].shape))
		for q in range(10):
			step = make_step(net, step)
		alldata += [step]
		# for j in range(10):
		# 	solver.step(1)

	save_wav("will_it_dream.wav", np.array(alldata))

if __name__ == "__main__":
	dream()