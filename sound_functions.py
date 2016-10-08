import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint
import sys

rate = 44100



def get_raw(filename, seconds, label, max_out = 0, width=400):

    allthedata = get_all_the_data(filename, seconds, max_out, width)
    thelabel = np.array([label])
    labeldata = []
    rawdata = []
    for data in allthedata:

        rawdata += [data]
        labeldata.append(label)
    return [rawdata, labeldata]




def get_all_the_data(filename, seconds, max_out, width):

    data = wav.read(filename)

    
    data = np.array(data[1], dtype=np.float32)
    print "indata shape:"
    print data.shape
    data = data/(2*np.max(data))


    # Convert to mono
    # if(data.ndim > 1):
    #     data = np.sum(data, axis=1)
    # print data.shape

    data = np.swapaxes(data, 0, 1) # caffe wants channel to be first axis


    things = int(float(len(data[0]))/(rate*seconds))
    
    if(max_out > 0):
        things = min(things, max_out)
    print "things: " + str(things)
    m = seconds*rate
    data = data[:,0:int(m*things)]
    data = data.reshape((2, data.size/(2*width), width))
    print data.shape
    allthedata = []
    d = data.shape[1]/things
    print "d:"
    print d
    for i in range(0, things):
        l = int(d*i)
        h = int(d*(i+1))
        dataslice = data[:, l:h, : ]

       
    

        allthedata += [dataslice]
    return allthedata



def save_raw(filename, allthedata):

    allthedata = np.array(allthedata)
    print "allthedata shape:"
    print allthedata.shape
    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]*allthedata.shape[2], 2), dtype=np.float32)

    for i in range(allthedata.shape[0]):
        data = allthedata[i]
       
        data = data.reshape((data.shape[0], data.size/data.shape[0]))
        data = np.swapaxes(data, 0, 1)
        #data = data*20000.0
     
       
        outdata [i][:] = np.copy(data[:])


    outdata = outdata.reshape((outdata.size/2, 2))
  
    outdata = outdata.astype(np.int16)

    z = np.zeros(outdata.shape, dtype=np.int16)
    outdata = np.append(outdata, z)
    print "outdata shape:"
    print outdata.shape
    print "outdata:"
    print outdata[0:20] 
    # Write to disk
    rate = 44100
    output_file = wave.open(filename, "w")
    output_file.setparams((2, 2, rate, 0, "NONE", "not compressed"))

    output_file.writeframesraw(outdata)
    output_file.close()    

def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])


def time_to_shape(seconds, width):
    return [int(rate*seconds/width), width]





def test_save_wav():
    filename = '../songsinmyhead/08dreams.wav'
    [fftdata, fftlabels] = get_raw(filename, 5, 19, width=300, max_out=0)
    print 'max, min, avg:'
    print np.max(fftdata)
    print np.min(fftdata)
    print np.average(fftdata)
    save_raw('test3.wav', fftdata)
    print("saved test3.wav")

if __name__ == "__main__":
    test_save_wav()