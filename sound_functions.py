import scipy.io.wavfile as wav
import wave
import numpy as np
import math
from pprint import pprint

rate = 44100
def get_fft(filename, seconds, label, max_out = 0):

    # Read in
    data = wav.read(filename)
    
    fftwidth = rate*2
    data = np.array(data[1], dtype=np.float32)
    
    #data = 255.0*data.astype(np.float32)/(np.max(data)*2.0)
    #print(np.max(data))
    # Convert to mono
    if(data.ndim > 1):
        data = np.sum(data, axis=1)
    print data.shape

    things = int(float(len(data))/(rate*seconds*2))
    
    if(max_out > 0):
        things = min(things, max_out)
    print "things: " + str(things)
    m = seconds*rate*2
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
    #labels = np.zeros((len(allthedata[0])/fftwidth, fftwidth))
    #print labels.shape
    #labels[0][label] = 1.0
    thelabel = np.array([label])

    labeldata = []
    for data in allthedata:
        #data.resize( len(data)/fftwidth, fftwidth) TMPDEBUG

        # Get fft and separate out magnitude and phase
        #data = np.fft.fft(data, fftwidth, axis=1) TMPDEBUG
        data = np.fft.fft(data)
        print "data in.shape:"
        print data.shape
        mag= np.absolute(data)
        #mag  = 255*mag/ (np.max(mag))
        phase = np.angle(data)
        #phase = phase + 2*np.pi
       


        mp = np.concatenate((mag, phase))
    
        print(mp.shape)
        fftdata += [[[mp]]]
        labeldata.append(label)
        


    return [fftdata, labeldata]



def save_wav(filename, allthedata):
    print("allthedata:")

    allthedata = np.array(allthedata)
    print allthedata.shape
    #outdata = np.ndarray((allthedata.shape[0], allthedata.shape[3]/2), dtype=np.int16)
    outdata = []
    for i in range(allthedata.shape[0]):
        data = allthedata[i]
        data = data[0][0]
        data = np.array(data)
        mag = data[:data.shape[0]/2]
        phase = data[data.shape[0]/2:]
        


        #mag = mag*8000
        data = mag_phase_to_complex(mag, phase)


        #convert back to sound data
        print "save_wav data.shape:"
        print data.shape
        #data = np.fft.irfft(data, data.shape[1], axis=1) #TMPDEBUG
        data = np.fft.irfft(data)
        
        data = 0.75*data

        # # Put data back in the right format
        #data = np.reshape(data, (data.shape[0]*data.shape[1])) #TMPDEBUG
        data = data/2.0
        #data = 30000*data
        data = np.int16(data)
        print("data:")
        pprint(data)
        print data.shape
        outdata += [data]
        
    
    print("outdata:")
    pprint(outdata)
    print "final data length: "
    mydata = np.ndarray((1), dtype=np.int16)
    for data in outdata:
        mydata = np.append(mydata, data)
    
    #mydata = np.reshape(outdata, (outdata.shape[0]*outdata.shape[1]))

    print mydata.shape

    #monodata = np.append(mydata, np.zeros(mydata.shape, dtype=np.int16))
    #pprint(mydata)
    
    # Write to disk
    rate = 44100
    output_file = wave.open(filename, "w")
    output_file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    output_file.writeframes(mydata)
    output_file.close()



def mag_phase_to_complex(mag, phase):
    return np.array([mag[p]*np.exp(1.j*phase[p]) for p in range(0,len(mag))])


def test_save_wav():
    filename = '../sounds/04lovebites.wav'
    [fftdata, fftlabels] = get_fft(filename, 1, 3)
    save_wav('test3.wav', [fftdata[0]])
    print("saved wav")

#test_save_wav()