import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint

rate = 44100
bytes_per_sample = 2
def get_fft(filename, seconds, label, max_out = 0):

    # Read in
    data = wav.read(filename)
    
    fftwidth = rate/30
    data = data[1]
    data = 255.0*data.astype(np.float32)/(np.max(data)*2.0)
    #print(np.max(data))
    # Convert to mono
    if(data.ndim > 1):
        data = np.sum(data, axis=1)
    print data.shape

    things = int(len(data)/(rate*seconds*bytes_per_sample))
    if(max_out > 0):
        things = min(things, max_out)
    print("things: " + str(things))
    data = data[0:int(rate*seconds*things)]
    allthedata = [data[start:int(start + seconds*rate*bytes_per_sample)] for start in range(0, things) ]
    
    print "label = " + str(label)
    fftdata = []
    #labels = np.zeros((len(allthedata[0])/fftwidth, fftwidth))
    #print labels.shape
    #labels[0][label] = 1.0
    thelabel = np.array([label])

    labeldata = []
    for data in allthedata:
        data.resize( len(data)/fftwidth, fftwidth)

        # Get fft and separate out magnitude and phase
        data = np.fft.fft(data, fftwidth, axis=1)
        mag= np.absolute(data)
        mag = 255*(mag/ (np.max(mag)*2))
        
        phase = np.angle(data)
        phase = phase + 2*np.pi
        phase = 255*(phase/(np.pi*2))
        #print("change:")
        #print(phase.shape)
        mp = np.concatenate((mag, phase))
        #print(mp.shape)
        fftdata += [[mp]]
        labeldata.append(label)
        


    return [fftdata, labeldata]



def save_wav(filename, allthedata):
    print("allthedata:")

    allthedata = np.array(allthedata)
    print allthedata.shape
    outdata = np.ndarray((allthedata.shape[0], allthedata.shape[2]*allthedata.shape[3]/2), dtype=np.int16)
    for i in range(allthedata.shape[0]):
        data = allthedata[i]
        data = data[0]
        data = np.array(data)
        mag = data[:data.shape[0]/2]
        phase = data[data.shape[0]/2:]
        phase = 2.0*np.pi*phase/255.0

        mag = mag*2000
        data = mag_phase_to_complex(mag, phase)


        #convert back to sound data
        data = np.fft.irfft(data, data.shape[1], axis=1)


        # # Put data back in the right format
        data = np.reshape(data, (data.shape[0]*data.shape[1]))
        #data = data/2.0
        #data = 30000*data
        data = np.int16(data)
        print("data:")
        pprint(data)
        print data.shape
        outdata[i][:] = data
        
    
    print("outdata:")
    print "final data length: "
    
    
    outdata = np.reshape(outdata, (outdata.size))
    print outdata.shape
    pprint(outdata)
    # Write to disk
    rate = 44100
    output_file = wave.open(filename, "w")
    output_file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    output_file.writeframes(outdata)
    output_file.close()



def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])


def test_save_wav():
    filename = '../sounds/04lovebites.wav'
    [fftdata, fftlabels] = get_fft(filename, 1, 3)
    save_wav('test3.wav', [fftdata[0]])
    print("saved wav")

#test_save_wav()