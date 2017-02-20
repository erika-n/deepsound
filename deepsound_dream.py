
import numpy as np
from pprint import pprint
from shutil import copyfile
import getopt
import pickle
from deepsound_load import process_file, get_fft,  load_wav, save_dream_wav

import sys
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

song = '/home/erika/Music/songsinmyhead/d/88mysteriousways (2016_11_06 21_28_43 UTC).wav'
steps =10
skip = 1
run_name = 'raw_conv_relu'
end = 'fc1'
with open(run_name + '.pickle') as f:  
	params = pickle.load(f)

folder = params['folder']
   
batch_size = params['batch_size']
data_dim = params['data_dim'] # how many samples to look at 
test_instances = params['test_instances'] # number of tests for the test phase
training_instances = params['training_instances'] # training phase instances from each sound file.
fft = params['fft']
raw2d = params['raw2d']
raw2d_multiplier = params['raw2d_multiplier']
print "data_dim: "
print data_dim



model_def = 'soundnet/' + run_name + '_deploy.prototxt'
model_weights = 'soundnet/' + run_name + '.caffemodel'
solver_file = 'soundnet/smallsolver.prototxt'
restore_file = 'soundnet/' + run_name + '.solverstate'

outfile = "will_it_dream.wav"

def objective_L2(dst):
	dst.diff[:] = dst.data
	#dst.data[0][:] = dst.data[0]*1e-6


def get_weight_vector(net, i):
	v = net.params.items()[0][1][0].data[i]
	print "v:"
	print v.shape
	pprint(v)
	return v*10e7


def make_step(net, mydata, step_size=1000, end='score',
	 objective=objective_L2, label=None, bias=None):
	'''Basic gradient ascent step.'''

	# if bias: # set bias of first layer to zero except for the given one-- an experiment.
	# 	net.params.items()[0][1][1].data[:] = 0
	# 	#net.params.items()[0][1][1].data[bias] = 1

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
		dst.diff[:] = 0
		for l in label:
			dst.diff[0][l] = 100




	net.backward(start=end) 


	g = src.diff[0]

	print "g:"
	pprint(g)
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

	return (otherdata.copy(), dst.data.copy())
	

def dream():
	print "dreaming..."
	caffe.set_mode_cpu()


	net = caffe.Net(model_def, model_weights, caffe.TRAIN)     



	input_data = process_file(song, data_dim, 500, fft=fft, raw2d=raw2d, raw2d_multiplier=raw2d_multiplier)



	code = int('08')
	input_labels = np.array([code]*len(input_data))
	input_labels = input_labels.resize((input_labels.size, 1, 1, 1))

	input_data = np.array(input_data, dtype=np.float32)

	song_wav = load_wav(song)
	(mag, seed_phase) = get_fft(song_wav[10000:10000+data_dim[1]])
	
	print "input_data:"
	print input_data.shape
	print net.blobs['data'].data.shape

	net.blobs['data'].data[0][:] = input_data[0]
	net.forward()

	print 'predicted class is: ' + str(net.blobs['score'].data.argmax(1))
	#print 'real class is: ' + str(input_labels[0])


	# alldata = []
	# for i in range(10):
		
	# 	arr = np.array(get_weight_vector(net, i))
	# 	arr = arr.reshape((1, data_dim[0], data_dim[1]))

	# 	alldata += [arr]


	seed = input_data[100]

	
	seed = 10*(np.random.random_sample(seed.shape) - 0.5) #this seems like an interesting thing to do
	#seed = np.zeros(seed.shape)
	seed = 0.8*seed
	print "seed:"
	print seed.shape
	print seed[0][100:110]

	alldata = [seed]

	
	l = 0

	(step, outa) = make_step(net, seed, end=end)

	
	for i in range(steps):

		print "step " + str(i)
		if(i %skip == 0):
			
			
		
			alldata += [ step[0]]
		for j in range(1):
			(step, outa) = make_step(net, step, end=end, label=[2]) 
		l = l + 1


	print "alldata.shape:"
	print np.array(alldata[1]).shape
	
	save_dream_wav(alldata, step.shape, outfile, fft=fft, raw2d=raw2d, raw2d_multiplier=raw2d_multiplier, seed_phase=None)



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