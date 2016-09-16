import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint


def get_fft(filename, seconds, label, max_out = 0):

    # Read in
    data = wav.read(filename)
    rate = data[0]
    fftwidth = rate/30
    data = data[1]
    #data = 255.0*data.astype(np.float32)/(np.max(data)*2.0)
    #print(np.max(data))
    # Convert to mono
    if(data.ndim > 1):
        data = np.sum(data, axis=1)
    print data.shape

    things = int(len(data)/(rate*seconds))
    if(max_out > 0):
        things = min(things, max_out)
    print("things: " + str(things))
    data = data[0:int(rate*seconds*things)]
    allthedata = [data[start:int(start + seconds*rate)] for start in range(0, things) ]
    
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
        #phase = 255*(phase/(np.max(phase)*2))
        #print("change:")
        #print(phase.shape)
        mp = np.concatenate((mag, phase))
        #print(mp.shape)
        fftdata += [[mp]]
        labeldata.append(label)
        


    return [fftdata, labeldata]



def save_wav(filename, data):
    pprint(data)
    mag = data[:data.shape[0]/2]
    phase = data[data.shape[0]/2:]
    print data.shape
    print "mag, phase shape:"
    print mag.shape
    print phase.shape
    pprint(mag)

    mag = mag*20000
    data = mag_phase_to_complex(mag, phase)


    #convert back to sound data
    data = np.fft.irfft(data, data.shape[1], axis=1)


    # # Put data back in the right format
    data = np.reshape(data, (data.shape[0]*data.shape[1]))

    data = np.int16(data)

    # Write to disk
    rate = 44100
    output_file = wave.open(filename, "w")
    output_file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    output_file.writeframes(data)
    output_file.close()



def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])


def test_save_wav():
    filename = '../sounds/03orangecrush.wav'
    [fftdata, fftlabels] = get_fft(filename, 4, 3, 2)
    save_wav('test3.wav', np.array(fftdata[0][0]))
    print("saved wav")
