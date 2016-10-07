from sound_functions import get_fft, get_raw
from sound_functions import save_wav, save_raw
import numpy as np
from pprint import pprint
from shutil import copyfile

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

song = '../songsinmyhead/08dreams.wav'
label = 8
seconds = 1
frames_per_second = 100
model_def = 'soundnet/fiftypercent_deploy.prototxt'
model_weights = 'soundnet/small_iter_10000.caffemodel'
solver_file = 'soundnet/smallsolver.prototxt'
restore_file = 'soundnet/small_iter_2500.solverstate'

def objective_L2(dst):
	dst.diff[:] = dst.data 
	

def make_step(net, mydata, step_size=1000, end='conv1',
	jitter=4, clip=True, objective=objective_L2, label=None):
	'''Basic gradient ascent step.'''

	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]


	
	src.data[0][:] = mydata




	net.forward(end=end)
	# print "dst.data:"
	# pprint(dst.data)
	

	objective(dst)  # specify the optimization objective 


	if label:
		dst.diff[0][label] = 100

	net.backward(start=end) 


	g = src.diff[0]

	# print "g: "
	# pprint(g)

	# normalized ascent step 
	ascent = step_size/np.abs(g).mean() * g

	print "ascent: "
	pprint(ascent)

	

	otherdata = ascent 
	# print "otherdata: "
	# pprint (otherdata)

	return otherdata.copy()
	

def dream():
	print "dreaming..."
	caffe.set_mode_cpu()


	#model_weights = 'soundnet/simplenet3_5_500.caffemodel'
	#input_dim: 16
	#input_dim: 22050

	net = caffe.Net(model_def, model_weights, caffe.TRAIN)     

	[a_song_data, a_song_labels] = get_raw(song, seconds, label, 200, frames_per_second=frames_per_second)



	input_data = np.array(a_song_data, dtype=np.float32)
	input_labels = np.array(a_song_labels, dtype=np.float32)
	print input_data.shape
	print net.blobs['data'].data.shape


	net.blobs['data'].data[0][:] = input_data[0]
	#net.blobs['label'].data[0] = input_labels
	net.forward()

	print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
	print 'real class is: ' + str(input_labels[0])


	duckunder = 1#10000

	alldata = [input_data[0]*1000]	
	step = make_step(net,input_data[15])

	for i in range(25):
		step = make_step(net, step)
		if(i %1== 0):
			
			print "step " + str(i)
			alldata += [ step ]

		



	save_raw("will_it_dream.wav", np.array(alldata))



if __name__ == "__main__":
	dream()