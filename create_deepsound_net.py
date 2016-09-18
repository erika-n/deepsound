
from pprint import pprint
import os
from os import listdir
from os.path import isfile, join

import numpy as np

import scipy.io.wavfile as wav
import wave

from pylab import *

# * Import `caffe`, adding it to `sys.path` if needed. Make sure you've built pycaffe.

solver_file = 'soundnet/simplesolver.prototxt'
caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

os.environ["GLOG_minloglevel"] = "4"

from caffe import layers as L, params as P

from sound_functions import get_fft


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def preprocess_data():

    folder = '../sounds/'
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    input_data = []
    input_labels = []
    test_data = []
    test_labels = []
    batch_size = 1
    numtests = 2
    test_seconds = 0.5
    input_data = []
    tmp_input_labels = []
    num_per_song = 10
    for i in range(len(files)):
        label = int(files[i][:2])
        [nextfft, nextlabel] = get_fft(folder + '/' + files[i], test_seconds, label, num_per_song)
        input_data += nextfft[1:]
        tmp_input_labels += [label for i in range(len(nextfft) - 1)]
        test_data += [nextfft[0]]
        test_labels += [label]




    input_data = np.array(input_data)
    input_labels = np.ndarray(len(tmp_input_labels))
    for i in range(len(tmp_input_labels)):
        input_labels[i]= tmp_input_labels[i]


    #meaninput = np.average(input_data, axis=0)
    #meaninput = meaninput.reshape((input_data.shape))
    #print meaninput.shape

    #print input_data.shape
    #input_data = input_data - meaninput + 0.5
    #test_data = test_data - meaninput + 0.5

    input_labels = np.ascontiguousarray(input_labels[:, np.newaxis, np.newaxis,
                                                 np.newaxis])


    #input_data = np.copy(input_data[15:])
    #input_labels = np.copy(input_labels[15:])
    total_size = len(input_data) - (len(input_data)%batch_size)
    print "total_size = " + str(total_size)




    #right now I don't have enough data so making lots more to shuffle.


    

    [input_data, input_labels] = shuffle_in_unison_inplace(input_data, input_labels)

    print "max, min, average: "
    print np.max(input_data)
    print np.min(input_data)
    print np.average(input_data)

    
    test_data = np.array(test_data, dtype=np.float32)
    input_data = np.array(input_data, dtype=np.float32)

    test_labels = np.array(test_labels, dtype=np.float32)
    input_labels = np.array(input_labels, dtype=np.float32)



    print "data, labels shapes: "
    print input_data.shape
    print input_labels.shape
    print "test, test labels shapes:"
    print test_data.shape
    print test_labels.shape

    np.save('preprocessed_sound.npy', [input_data, input_labels, test_data, test_labels])
    return [input_data, input_labels, test_data, test_labels]


process_data = True
if(process_data):
    [input_data, input_labels, test_data, test_labels] = preprocess_data()
else:
    [input_data, input_labels, test_data, test_labels] = np.load('preprocessed_sound.npy')


# would be nice to use this but not using it right now.
def lenet(batch_size):
    

    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data = L.MemoryData(batch_size=batch_size, channels=2, height=30, width=1470)
    #n.label = L.MemoryData(batch_size=batch_size, channels=2, height=30, width=1470)
    #n.label = L.Data(batch_size=batch_size, backend=P.Data.MemoryData, source=label_arr)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()
    
# with open('deepsound_train_mem.prototxt', 'w') as f:
#     f.write(str(lenet(12)))
    
# with open('deepsound_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet(12)))

# gpu mode?
#caffe.set_device(0)
#caffe.set_mode_gpu()


### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(solver_file)
net = solver.net

print "setting input arrays. input_data.shape:"
print input_data.shape

net.set_input_arrays(input_data, input_labels)

testnet = solver.test_nets[0]
testnet.set_input_arrays(test_data, test_labels)
# * To get an idea of the architecture of our net, we can check the dimensions of the intermediate features (blobs) and parameters (these will also be useful to refer to when manipulating data later).

# In[8]:

# each output is (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]


# In[9]:

# just print the weight sizes (we'll omit the biases)
print[(k, v[0].data.shape) for k, v in solver.net.params.items()]


# * Before taking off, let's check that everything is loaded as we expect. We'll run a forward pass on the train and test nets and check that they contain our data.


solver.net.forward()  # train net

solver.test_nets[0].forward()  # test net (there can be more than one)


# In[11]:


print 'train labels:', solver.net.blobs['label'].data[:]
print 'test labels:', solver.test_nets[0].blobs['label'].data[:]

imshow(solver.net.blobs['data'].data[0][0], cmap='gray'); axis('off')
niter = 1000
test_interval = 1
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 10, 30))

# the main solver loop
for it in range(niter):
    
    print "solving, iteration = " + str(it)
    # store the train loss
    #train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batc
    # (start the forward pass at conv1 to avoid loading new data)
    #solver.test_nets[0].forward(start='conv1')
    #output[it] = solver.test_nets[0].blobs['score'].data[:30]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it == 0 or it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        for test_it in range(5):
            solver.test_nets[0].forward()
            # for lenet:
            print "hypothesis: " + str(solver.test_nets[0].blobs['score'].data.argmax(1))
            print "actual: " + str(solver.test_nets[0].blobs['label'].data)
            print "loss: " + str(solver.test_nets[0].blobs['loss'].data)
            # for google net:
            # print "hypothesis: " + str(solver.test_nets[0].blobs['loss1/classifier'].data.argmax(1))
            # print "top1:" + str(solver.test_nets[0].blobs['loss1/top-1'].data)
            # print "actual: " + str(solver.test_nets[0].blobs['label'].data)
            # print "loss: " + str(solver.test_nets[0].blobs['loss1/loss1'].data) 




            #pprint(solver.test_nets[0].blobs['conv1/7x7_s2'].data[0][0:20])           

    
    solver.step(1)  # SGD by Caffe

# In[16]:
sys.exit()
