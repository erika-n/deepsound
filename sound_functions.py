import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint
import sys

rate = 44100





def get_fft(filename, seconds, label, max_out = 0, frames_per_second = 30):
    fftwidth = rate/frames_per_second
    # Read in
    data = wav.read(filename)
    
    
    data = np.array(data[1], dtype=np.float32)
    #data = 255.0*data.astype(np.float32)/(np.max(data)*2.0)
    #print(np.max(data))
    # Convert to mono
    if(data.ndim > 1):
        data = np.sum(data, axis=1)
    print data.shape
    
    things = int(float(len(data))/(rate*seconds))
    
    if(max_out > 0):
        things = min(things, max_out)
    print "things: " + str(things)
    m = seconds*rate
    data = data[0:int(m*things)]
    
    allthedata = []
    for i in range(0, things):
        l = int(m*i)
        h = int(m*(i+1))
        dataslice = data[l:h ]
        allthedata += [dataslice]
    print np.array(allthedata).shape
    print "label = " + str(label)
    fftdata = []
   
    thelabel = np.array([label])
    labeldata = []
    for data in allthedata:
        
        if(fftwidth > 1):
            data.resize( len(data)/fftwidth, fftwidth) 
            # Get fft and separate out magnitude and phase
            data = np.fft.fft(data, fftwidth, axis=1)            
        else:
            data = np.fft.fft(data)
        mag= np.absolute(data)

        
        phase = np.angle(data)
        phase = phase + np.pi
        
        # new experiment: put mag and phase in different channels. Use this with channel splitting. 
        # mp = np.empty((2, mag.shape[0], mag.shape[1]))
        # mp[0][:] = mag
        # mp[1][:] = phase
        
        mp = np.concatenate((mag, phase)) 


        if fftwidth > 1:
            fftdata += [[mp]]
        else:
            fftdata += [mp]
        labeldata.append(label)
        
    return [fftdata, labeldata]



def save_wav(filename, allthedata):
    print("allthedata:")
    allthedata = np.array(allthedata)
    allthedata = np.array(allthedata)
    #allthedata = np.average(allthedata) + allthedata #TMPDEBUG undo negatives added for ReLU
    mydata = []
    print allthedata.shape
  
    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]*allthedata.shape[2]/2), dtype=np.int16)
    for i in range(allthedata.shape[0]):
        data = allthedata[i][0]

        # mag = data[0]

        # phase = data[1]

        mag = data[:data.shape[0]/2]
        mag = 0.0001*mag
        phase = data[data.shape[0]/2:]


        data = mag_phase_to_complex(mag, phase)
        #convert back to sound data
        print data.shape
        data = np.fft.irfft(data, data.shape[1], axis=1) 
        data = np.reshape(data, (data.shape[0]*data.shape[1])) 

        
        print "data: "
        pprint(data)
        # # Put data back in the right format
        

        data = 0.0005*data
        data = data.astype(np.int16)
        print data.shape
        print i
        print outdata.shape

        outdata [i][:] = np.copy(data[:])

        
    
    print("outdata:")
    pprint(outdata)

    monodata = np.append(outdata, np.zeros(outdata.shape, dtype=np.int16))

    print "outdata 2:"
    outdata = np.array(outdata, dtype=np.int16)
    outdata = outdata.reshape((outdata.shape[0]*outdata.shape[1]))  
    print outdata[0:20] 
    # Write to disk
    rate = 44100
    output_file = wave.open(filename, "w")
    output_file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    output_file.writeframes(monodata)
    output_file.close()

def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])

def time_to_shape(seconds, frames_per_second):
    fftwidth = rate/frames_per_second
    frames = 2*seconds*frames_per_second
    return [  int(frames), int(fftwidth)] 







def get_raw(filename, seconds, label, max_out = 0, width=400, channels=2):

    allthedata = get_all_the_data(filename, seconds, max_out, width, channels)
    thelabel = np.array([label])
    labeldata = []
    rawdata = []
    for data in allthedata:

        rawdata += [data]
        labeldata.append(label)
    return [rawdata, labeldata]




def get_all_the_data(filename, seconds, max_out, width, channels=2):

    data = wav.read(filename)

    
    data = np.array(data[1], dtype=np.float32)
    print "indata shape:"
    print data.shape
    data = data/(2*np.max(data))


    # Convert to mono if need be
    if(channels < 2 and data.ndim > 1):
        data = np.sum(data, axis=1)

    
    else:
        data = np.swapaxes(data, 0, 1) # caffe wants channel to be first axis
    print data.shape
    things = int(float(data.shape[0])/(rate*seconds))
    
    
    if(max_out > 0):
        things = min(things, max_out)
    print "things: " + str(things)
    m = seconds*rate
    if(channels > 1):
        data = data[:,0:int(m*things)]
    else:
        data = data[0:int(m*things)]
    data = data.reshape((channels, data.size/(channels*width), width))
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



def save_raw(filename, allthedata, channels=2):

    allthedata = np.array(allthedata)
    print "allthedata shape:"
    print allthedata.shape
    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]*allthedata.shape[2], 2), dtype=np.float32)

    for i in range(allthedata.shape[0]):
        data = allthedata[i]
       
        data = data.reshape((data.shape[0], data.size/data.shape[0]))
        if(channels == 2):
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
    output_file.setparams((channels, 2, rate, 0, "NONE", "not compressed"))

    output_file.writeframesraw(outdata)
    output_file.close()    

def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])


# def time_to_shape(seconds, width):
#     return [int(rate*seconds/width), width]





def test_save_wav():
    filename = '../songsinmyhead/08dreams.wav'
    [fftdata, fftlabels] = get_fft(filename, 5, 19, frames_per_second=60, max_out=0)
    print 'max, min, avg:'
    print np.max(fftdata)
    print np.min(fftdata)
    print np.average(fftdata)
    save_wav('test3.wav', fftdata)
    print("saved test3.wav")

if __name__ == "__main__":
    test_save_wav()