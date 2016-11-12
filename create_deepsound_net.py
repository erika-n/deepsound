
from pprint import pprint
import os
from os import listdir
from os.path import isfile, join

import numpy as np

import scipy.io.wavfile as wav
import wave
import sys, getopt

from pylab import *

import lmdb
from PIL import Image



folder = '../sounds/testsounds/'

batch_size = 1
numtests = 10
test_seconds = 0.5
rate = 44100
test_width = 10
length = rate*test_seconds
test_height = int(length/test_width)
test_frames_per_second = 30
num_per_song = 20


solver_file = 'soundnet/smallsolver.prototxt'

# * Import `caffe`, adding it to `sys.path` if needed. Make sure you've built pycaffe.


caffe_root = '/home/erika/projects/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

os.environ["GLOG_minloglevel"] = "4"

from caffe import layers as L, params as P

from sound_functions_2 import get_sound_data
from sound_functions import get_fft


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]






def preprocess_data():



    
    input_data = []
    input_labels = []
    test_data = []
    test_labels = []
    
    tmp_input_labels = []



    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for i in range(len(files)):
        label = int(files[i][:2])

        [fft, labels] = get_fft(folder + '/' + files[i], test_seconds, label, max_out=num_per_song, frames_per_second=test_frames_per_second)       
        input_data += fft
        #input_labels.append(labels)
        # sounddata = get_sound_data(folder + '/' + files[i])
        # for n in range(int(min((sounddata.shape[0]/length) - 2, num_per_song))):
        #     data = sounddata[n*length:(n+1)*length]
        #     print data.shape
        #     input_data += [data.reshape((1, test_height, test_width))]
        #     label = np.array([sounddata[(n+1)*length + 1]])
        #     input_labels += [label.reshape(1, 1,1)]




    print (len(input_data))
    input_data = np.array(input_data, dtype=np.float32)
    #input_labels = np.array(input_labels, dtype=np.float32)
    input_labels = np.copy(input_data)
    input_labels = np.roll(input_labels,1, axis=0)
    input_data, input_labels = shuffle_in_unison_inplace(input_data, input_labels)
    


    print "max, min, average: "
    print np.max(input_data)
    print np.min(input_data)
    print np.average(input_data)


    test_data = input_data[0:numtests]
    input_data = input_data[numtests:]

    test_labels = input_labels[0:numtests]
    input_labels = input_labels[numtests:]




    print "data, labels shapes: "
    print input_data.shape
    print input_labels.shape
    print "test, test labels shapes:"
    print test_data.shape
    print test_labels.shape
    print "making databases..."
    make_database(input_data, 'inputs_lmdb')
    make_database(input_labels, 'labels_lmdb')
    make_database(test_data, 'test_inputs_lmdb')
    make_database(test_labels, 'test_labels_lmdb')
    
    
    #np.save('preprocessed_sound.npy', [input_data, input_labels, test_data, test_labels])


def make_database(inputs, name):
    inputs = inputs.astype(float)
    in_db = lmdb.open(name, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            im_dat = caffe.io.array_to_datum(in_)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()



def soundnet(batch_size, shape, deploy=False, test=''):
    print shape
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    if(deploy):
        n.data = L.Input(shape=dict(dim=[1, 1, shape[0], shape[1]]))
    else:
        n.label = L.Data(batch_size=1, backend=P.Data.LMDB, source=(test + 'labels_lmdb'), transform_param=dict(scale=3e-7), ntop=1)
        n.data = L.Data(batch_size=1, backend=P.Data.LMDB, source=(test + 'inputs_lmdb'), transform_param=dict(scale=3e-7), ntop=1)    
    

  
    # n.conv1 = L.Convolution(n.data, kernel_size=6, num_output=30, weight_filler=dict(type='xavier'))
    # #n.relu1 = L.ReLU(n.conv1, in_place=True)
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # # n.conv2 = L.Convolution(n.pool1, kernel_size=4, num_output=10, weight_filler=dict(type='xavier'))
    # # #n.relu2 = L.ReLU(n.conv2, in_place=True)
    # # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    
    # #n.conv3 = L.Convolution(n.pool2, kernel_size=4, num_output=30, weight_filler=dict(type='xavier'))
    # # n.relu3= L.ReLU(n.conv3, in_place=True)
    # # n.pool3 = L.Pooling(n.relu3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # #n.conv4 = L.Convolution(n.pool3, kernel_size=2, num_output=30, weight_filler=dict(type='xavier'))
    # #n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # # n.conv5 = L.Convolution(n.pool4, kernel_size=3, num_output=30, weight_filler=dict(type='xavier'))
    # # n.pool5 = L.Pooling(n.conv5, kernel_size=3, stride=2, pool=P.Pooling.MAX)    


    

    # n.fc1 =   L.InnerProduct(n.pool1, num_output=30, weight_filler=dict(type='xavier'))

    # n.relu1 = L.ReLU(n.fc1, in_place=True)


    


    n.fc1 =  L.InnerProduct(n.data, num_output=1000, weight_filler=dict(type='xavier'))
    n.s1 = L.Sigmoid(n.fc1, in_place=True)
    n.fc2 =   L.InnerProduct(n.fc1, num_output=400, weight_filler=dict(type='xavier'))
    n.s2 = L.Sigmoid(n.fc2, in_place=True)
    #n.fc3 =   L.InnerProduct(n.fc2, num_output=300, weight_filler=dict(type='xavier'))
    n.score = L.InnerProduct(n.fc2, num_output=44100, weight_filler=dict(type='xavier'))
    
    # n.fc4 =   L.InnerProduct(n.fc3, num_output=500, weight_filler=dict(type='xavier'))
    # n.s4 = L.Sigmoid(n.fc4, in_place=True)
    # n.fc5 =   L.InnerProduct(n.fc4, num_output=400, weight_filler=dict(type='xavier'))
    # n.s5 = L.Sigmoid(n.fc5, in_place=True)

    # n.fc3 = L.InnerProduct(n.pool1, num_output = 200,  weight_filler=dict(type='xavier'))


    # n = caffe.NetSpec()
    
    # n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
    #                          transform_param=dict(scale=1./255), ntop=2)
    


    #lenet:
    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    # n.relu1 = L.ReLU(n.fc1, in_place=True)
    # n.score = L.InnerProduct(n.relu1, num_output=44100, weight_filler=dict(type='xavier'))

    

   
    
    if not deploy:
        n.loss =  L.EuclideanLoss(n.score, n.label)
 
    return  n.to_proto()
    




def main():


    # gpu mode?
    #caffe.set_device(0)    
    #caffe.set_mode_gpu()
    filename = 'soundnet/auto_small'
    

    with open(filename + '_train.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [test_height, test_width], False)))

    with open(filename + '_test.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [test_height, test_width], False, test='test_')))
    with open(filename + '_deploy.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [test_height, test_width], True)))
    

    print("writing " + filename)


    #[input_data, input_labels, test_data, test_labels] = np.load('preprocessed_sound.npy')
    ### load the solver and create train and test nets
    solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.SGDSolver(solver_file)
    #solver.restore('soundnet/phantom_1000.solverstate')

    # print "setting input arrays. input_data.shape:"
    # print input_data.shape
    
    net = solver.net
    # net.set_input_arrays(input_data, input_labels)

    testnet = solver.test_nets[0]
    # testnet.set_input_arrays(test_data, test_labels)


    # print out the shapes of the blobs in the net
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print[(k, v[0].data.shape) for k, v in solver.net.params.items()]
    print "TEST:"
        # print out the shapes of the blobs in the net
    print [(k, v.data.shape) for k, v in testnet.blobs.items()]
    print[(k, v[0].data.shape) for k, v in testnet.params.items()]


    # * Before taking off, let's check that everything is loaded as we expect. We'll run a forward pass on the train and test nets and check that they contain our data.

    solver.net.forward()  # train net
    solver.test_nets[0].forward()  # test net (there can be more than one)


    # In[11]:


    print 'train labels:', solver.net.blobs['label'].data[:]
    print 'test labels:', solver.test_nets[0].blobs['label'].data[:]


    niter = 10000
    test_interval = 10
    # losses will also be stored in the log

    int_tests = 5
    # the main solver loop
    for it in range(niter):
        
        print "solving, iteration = " + str(it)
 
        
        # run a full test every so often

        
        if it == 0 or it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            print "Training set: "
            test_net(solver.net, int_tests)
            print "Test set:"
            test_net(solver.test_nets[0], int_tests)
        
        solver.step(1)  # SGD by Caffe


def test_net(net, numtests):
    correct = 0
    answers = []
    loss = 0
    for test_it in range(numtests):
        net.forward()
        # for lenet:
        print "hypothesis: " + str(net.blobs['score'].data)
        print "actual: " + str(net.blobs['label'].data)
        print "loss: " + str(net.blobs['loss'].data)
        loss += abs(net.blobs['loss'].data)
        correct += sum(net.blobs['label'].data[0][0] == net.blobs['score'].data.argmax(1))
        answers.append(net.blobs['score'].data.argmax(1))
        # for google net:
        # print "hypothesis: " + str(solver.test_nets[0].blobs['loss1/classifier'].data.argmax(1))
        # print "top1:" + str(solver.test_nets[0].blobs['loss1/top-1'].data)
        # print "actual: " + str(solver.test_nets[0].blobs['label'].data)
        # print "loss: " + str(solver.test_nets[0].blobs['loss1/loss1'].data) 
        # correct += sum(solver.test_nets[0].blobs['label'].data[0][0] == solver.test_nets[0].blobs['loss1/top-1'].data)
        # answers.append(solver.test_nets[0].blobs['loss1/top-1'].data)
    print str(correct) + "/" + str(numtests)    
    print "unique: " 
    print np.unique(np.array(answers, dtype=np.int16))
    print "loss:"
    print loss/numtests
      




if __name__ == "__main__":

    try:
      opts, args = getopt.getopt(sys.argv[1:],"hp",["preprocess"])
    except getopt.GetoptError:
      print 'create_deepsound_net.py [--preprocess]'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'create_deepsound_net.py [--preprocess]'
         sys.exit()
      elif opt == '-p' or opt in ( "--preprocess"):
         preprocess_data()
    main()        