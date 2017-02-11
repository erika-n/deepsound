
from pprint import pprint
import os

import numpy as np

import scipy.io.wavfile as wav
import wave
import sys, getopt

from pylab import * # pip install matplotlib, apt-get install python-tk

import lmdb #apt-get install python-lmdb
from PIL import Image
from deepsound_load import prepare_data
import pickle





solver_file = 'soundnet/smallsolver.prototxt'


import caffe

os.environ["GLOG_minloglevel"] = "4"

from caffe import layers as L, params as P

run_name = 'autocorrelate'
run_description = 'self obsession'
folder = '/home/erika/Music/songsinmyhead/c'
height = 1
width = 1000 # how many samples to look at 
batch_size = 1 
num_tests= 10 # number of tests for the test phase
training_instances = 20# training phase instances from each sound file.
fft = False 
raw2d = False
raw2d_multiplier = 0.0008

def preprocess_data():
    data_params = {
        'training_instances': training_instances, 
        'data_dim': [height, width], 
        'folder': folder,
        'test_instances': num_tests,
        'fft': fft,
        'run_name': run_name,
        'run_description': run_description,
        'batch_size': batch_size,
        'raw2d': raw2d,
        'raw2d_multiplier': raw2d_multiplier
    }

    with open('soundnet/' + run_name + '.pickle', 'w') as f:  
        pickle.dump(data_params, f)

    prepare_data(run_name)




def soundnet(batch_size, shape, deploy=False, test=''):
    print shape

    n = caffe.NetSpec()
    
    if(deploy):
        n.data = L.Input(shape=dict(dim=[1, 1, shape[0], shape[1]]))
    else:
        n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=(test + 'labels_lmdb'), ntop=1)
        #n.clip = L.Input(shape=dict(dim=[shape[0], shape[1]]))
        n.data = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=(test + 'inputs_lmdb'), ntop=1, transform_param={'scale':1} )    

    # n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=6, weight_filler=dict(type='xavier'))
    # n.relu1 = L.ReLU(n.conv1, in_place=True)
    # n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=1, pool=P.Pooling.MAX)


    # n.conv2 = L.Convolution(n.pool1, kernel_size=3, num_output=3, weight_filler=dict(type='xavier'))
    # n.relu2 = L.ReLU(n.conv2, in_place=True)
    # n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
   

    # n.lstm = L.LSTM(n.data,n.clip)
    # n.lrn = L.LRN(n.data, lrn_param=dict(norm_region=1))
    # n.fc1 =  L.InnerProduct(n.data, num_output=200, weight_filler=dict(type='xavier'))
    


    n.fc1 = L.InnerProduct(n.data, num_output = width, weight_filler=dict(type='xavier'))
    n.s1 = L.TanH(n.fc1, in_place=True)


    # n.fc2 =  L.InnerProduct(n.s1, num_output=200, weight_filler=dict(type='xavier'))
    # n.s2 = L.TanH(n.fc2, in_place=True)    #n.s2 = L.Sigmoid(n.fc2, in_place=True)
    # n.fc3 =  L.InnerProduct(n.fc2, num_output=300, weight_filler=dict(type='xavier'))
    # n.s3 = L.Sigmoid(n.fc3, in_place=True)

    n.score = L.InnerProduct(n.fc1, num_output=width*height, weight_filler=dict(type='xavier'))
   
    if not deploy:
        n.loss =  L.EuclideanLoss(n.score, n.label, loss_weight=1.0)
 
    return  n.to_proto()
    




def main():


    # gpu mode
    caffe.set_device(0)   
    caffe.set_mode_gpu()

    filename = 'soundnet/auto_small'
    

    with open(filename + '_train.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [height, width], False)))

    with open(filename + '_test.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [height, width], False, test='test_')))
    with open(filename + '_deploy.prototxt', 'w') as f:
        f.write("force_backward : true\n" + str(soundnet(batch_size, [height, width], True)))
    

    print("writing " + filename)



    ### load the solver and create train and test nets
    solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.SGDSolver(solver_file)
    #solver.restore('soundnet/phantom_1000.solverstate')

    

    # print out the shapes of the blobs in the net
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print[(k, v[0].data.shape) for k, v in solver.net.params.items()]

    solver.net.forward()  # train net
    solver.test_nets[0].forward()  # test net (there can be more than one)


    # In[11]:


    print 'train labels:', solver.net.blobs['label'].data[:]
    print 'test labels:', solver.test_nets[0].blobs['label'].data[:]


    niter = 10000
    test_interval = 100


    int_tests = 1
    # the main solver loop
    for it in range(niter):
        
        #print "solving, iteration = " + str(it)
        solver.step(100)  # SGD by Caffe
        
        # run a full test every so often

        
        if it == 0 or it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            print "Training set: "
            test_it(solver.net, int_tests)
            print "Test set:"
            test_it(solver.test_nets[0], int_tests)
        
        


def test_it(net, numtests):
    correct = 0
    answers = []
    loss = 0
    for test_it in range(numtests):
        net.forward()
        # for lenet:
        print "h: " + str(net.blobs['score'].data[0][0:10]) #str(net.blobs['score'].data.argmax(1))
        print "a: " + str(net.blobs['label'].data[0][0][0][0:10]) #str(net.blobs['label'].data.flatten().astype(np.int16))
       
        # # print "loss: " + str(net.blobs['loss'].data)
        # loss = abs(net.blobs['loss'].data)
        # correct = sum(net.blobs['label'].data.flatten().astype(np.int16) == net.blobs['score'].data.argmax(1))
        # answers.append(net.blobs['score'].data.argmax(1))

        # print str(correct)  + "/" + str(batch_size) + " correct (" + str(100*(batch_size - correct)/batch_size) + " percent error rate)"    
        # print "unique: " 
        # print np.unique(np.array(answers, dtype=np.int16))
        # print "loss:"
        # print loss
      




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