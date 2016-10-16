from sound_functions import get_raw, get_fft
from sound_functions import save_raw, save_wav
import numpy as np
from pprint import pprint
from shutil import copyfile
import getopt

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

song = '../moresounds/11carpassing.wav'
steps = 20
skip = 1

end = 'conv1'
label = 8
seconds = 3
#width=200
frames_per_second = 30
channels = 1
model_def = 'soundnet/layercake_deploy.prototxt'
model_weights = 'soundnet/layercake_2400.caffemodel'
solver_file = 'soundnet/smallsolver.prototxt'
restore_file = 'soundnet/small_iter_2500.solverstate'

outfile = "will_it_dream.wav"

def objective_L2(dst):
	dst.diff[:] = dst.data 

def make_step(net, mydata, step_size=200000, end='score',
	 objective=objective_L2, label=None):
	'''Basic gradient ascent step.'''

	
	src = net.blobs['data'] # input image is stored in Net's 'data' blob
	dst = net.blobs[end]


	
	src.data[0][:] = 0.01*mydata



	net.forward(end=end)
	if(end == 'score'):
		print 'in ascent, predicted class is: ' + str(net.blobs['score'].data.argmax(1))
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
	
	#ascent[0, :] = ascent[0]
	print "ascent: "
	print ascent[0, :].shape
	pprint(ascent)

	

	otherdata = ascent 
	# print "otherdata: "
	# pprint (otherdata)

	return otherdata.copy()
	

def dream():
	print "dreaming..."
	caffe.set_mode_cpu()


	net = caffe.Net(model_def, model_weights, caffe.TRAIN)     

	[a_song_data, a_song_labels] = get_fft(song, seconds, label, frames_per_second=frames_per_second)



	input_data = np.array(a_song_data, dtype=np.float32)
	input_labels = np.array(a_song_labels, dtype=np.float32)
	print input_data.shape
	print net.blobs['data'].data.shape


	net.blobs['data'].data[0][:] = input_data[5]

	net.forward()

	print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
	print 'real class is: ' + str(input_labels[0])

	seed = np.zeros(input_data[-10].shape)
	alldata = [seed]	

	
	l = 0
	step = make_step(net, seed, end=end, label=l)

	for i in range(steps):

		

		if(i %skip == 0):
			
			print "step " + str(i)

			alldata += [ step ]
		step = make_step(net, seed, end=end, label=l)
		l = l + 1

		



	save_wav(outfile, np.array(alldata))



if __name__ == "__main__":
    # try:
	opts, args = getopt.getopt(sys.argv[1:],"s:o:e:i:", ["steps=","output=","end=", "input=", "skip="])
    # except getopt.GetoptError:
    #   	print 'create_deepsound_net.py -s [steps] -o [output file]'
    #   	sys.exit(2)
	for opt, arg in opts:
		if opt in ('-s', '--steps'):
			steps = int(arg)
		elif opt in ('-o', '--output'):
			outfile = arg
		elif opt in('-e', '--end'):
			end = arg
		elif opt in('-i', '--input'):
			song = arg
		elif opt in('--skip'):
			skip = int(arg)
	dream()