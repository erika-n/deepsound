import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint
import sys

rate = 44100


# factors to fudge mag by to make it fit into 0-1
f1 = 255.0
f2 = 3
f3 = 5
f4 = 1
f5 = 1

def get_fft(filename, seconds, label, max_out = 0, frames_per_second = 30):

    fftwidth = rate/frames_per_second
    # Read in


    allthedata = get_all_the_data(filename, seconds, max_out)


    fftdata = []

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

        
   
        #experiment: interleave mag and phase.
        #mp = np.empty((mag.size + phase.size,), dtype = np.float32) #interleave mag and phase
        #mp[0::2] = mag.reshape(mag.size)
        #mp[1::2] = phase.reshape(phase.size)
        #mp = mp.reshape((mp.size/fftwidth, fftwidth))
       
 
        mp = np.concatenate((mag, phase)) 
        mp *= f5
        
        if fftwidth > 1:
            fftdata += [[mp]]
        else:
            fftdata += [[[mp]]]
        labeldata.append(label)
        


    return [fftdata, labeldata]


def get_raw(filename, seconds, label, max_out = 0, frames_per_second = 30):
    fftwidth = time_to_shape(seconds, frames_per_second)[1]
    allthedata = get_all_the_data(filename, seconds, max_out)
    thelabel = np.array([label])
    labeldata = []
    rawdata = []
    for data in allthedata:
        data.resize(len(data)/fftwidth, fftwidth)
      
        rawdata += [[data]]
        labeldata.append(label)
    return [rawdata, labeldata]




def get_all_the_data(filename, seconds, max_out):

    data = wav.read(filename)
    
    
    data = np.array(data[1], dtype=np.float32)
    data = data/(2*np.max(data))

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
    return allthedata



def save_wav(filename, allthedata):
    print("allthedata:")

    allthedata = np.array(allthedata)
    allthedata = np.array(allthedata)
    #allthedata = np.average(allthedata) + allthedata #TMPDEBUG undo negatives added for ReLU
    mydata = []
    print allthedata.shape

    is1d = False #(allthedata.shape[2] == 1)
    
    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]*allthedata.shape[2]/2), dtype=np.int16)

    for i in range(allthedata.shape[0]):
        data = allthedata[i]
        if not is1d:
            data = data[0]
        else:
            data = data[0][0]


        data /= f5
	    

        #experimental: combine mag and phase.
        # datawidth = data.shape[1]
        # data = data.reshape((data.size,))

        # mag = np.empty((data.size/2), dtype=np.float32)
        # phase = np.empty((data.size/2), dtype = np.float32)
        # #data /= 0.00001
        # mag[:] = data[0::2]
        # phase[:] = data[1::2]
        
        
        #phase = phase*2*np.pi
        #print "phase: "
        #pprint(phase)
        mag = data[:data.shape[0]/2]
        phase = data[data.shape[0]/2:]
        data = mag_phase_to_complex(mag, phase)
        # data = data.reshape((data.size/datawidth, datawidth)) #EXPERIMENTAL
        #convert back to sound data

        if not is1d:
            data = np.fft.irfft(data, data.shape[1], axis=1) 
            data = np.reshape(data, (data.shape[0]*data.shape[1])) 
        else:
            data = np.fft.ifft(data)
        
  
  
 
        data = 0.1*data
        data = data.astype(np.int16)

        print data.shape
        print i
        print outdata.shape
        #mydata = data
        if not is1d:
            outdata [i][:] = np.copy(data[:])
        else:
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

def save_raw(filename, allthedata):

    allthedata = np.array(allthedata)

    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]*allthedata.shape[2]), dtype=np.int16)

    for i in range(allthedata.shape[0]):
        data = allthedata[i][0]
        data = np.reshape(data, (data.shape[0]*data.shape[1]))
 
        data = data.astype(np.int16)

        outdata [i][:] = np.copy(data[:])

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
    frames = seconds*frames_per_second #EXPERIMENTAL: for raw only. 2*seconds*frames_per_second
    return [int(frames), int(fftwidth)]





def test_save_wav():
    filename = '../sounds/19lovebites.wav'
    [fftdata, fftlabels] = get_raw(filename, 1, 19, 10)
    print 'max, min, avg:'
    print np.max(fftdata)
    print np.min(fftdata)
    print np.average(fftdata)
    save_raw('test3.wav', fftdata)
    print("saved test3.wav")

if __name__ == "__main__":
    test_save_wav()