import scipy.io.wavfile as wav
import wave
import numpy as np


def get_fft(filename, seconds, label):

    # Read in
    data = wav.read(filename)
    rate = data[0]
    fftwidth = rate/30
    data = data[1]
    data = 255.0*data.astype(np.float32)/(np.max(data)*2.0)
    print(np.max(data))
    # Convert to mono
    if(data.ndim > 1):
        data = np.sum(data, axis=1)
    print data.shape

    things = int(len(data)/(rate*seconds))
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
        phase = 255*(phase/(np.max(phase)*2))
        
        fftdata += [[mag]]
        labeldata.append(label)
        


    return [fftdata, labeldata]

