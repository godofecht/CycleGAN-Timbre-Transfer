import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display as display

#to play audio
import IPython.display as ipd 
import numpy
from numpy import asarray
from numpy import save
import sys
import scipy
    
    
PADDED_WIDTH = 256
PADDED_HEIGHT = 256
SPEC_WIDTH = 256
SPEC_HEIGHT = 126

numpy.set_printoptions(threshold=sys.maxsize)
 
def display_spectrogram(spec,title):
  librosa.display.specshow(librosa.amplitude_to_db(spec, ref=numpy.max),sr=16000, x_axis='time', y_axis='cqt_note')
  plt.colorbar(format='%+2.0f dB')
  plt.title(title) 
  plt.tight_layout()
  plt.show()
 # plt.close()


def ConvertFolderToSpectrograms(input_dir,output_dir):
    file_count = 0    
    for filename in os.listdir(input_dir):
        x, sr = librosa.load(input_dir+'/'+filename, sr=None)
        
        #x = x/numpy.max(numpy.abs(x))*1
        
        

        #print(sr)

        #plt.figure(figsize=(14, 5))
        #display.waveplot(x, sr=sr)


        #X = numpy.abs(librosa.core.hybrid_cqt(x,sr = 16000))
        
        
        #X = numpy.abs(librosa.core.cqt(x,sr=16000))#,sr = 16000,hop_length = 512,bins_per_octave = int(12),n_bins=int(84)))
        #X = numpy.abs(librosa.stft(x,hop_length=128,n_fft = 1024))
        X = numpy.abs(librosa.feature.melspectrogram(y=x, sr=sr, n_mels=256,fmax=8000))
        
        #print(numpy.shape(X))
        X = X.astype(float)
        
        #Xlog = numpy.log(X + 1e-9)
        #X = librosa.util.normalize(Xlog) #normalized to -1 and 0
        
        #X = (((X+1)*2)-1) #normalize to -1 and 1
        print(numpy.shape(X))
        
        
        
        
        #X = numpy.resize(X,(256,256))
        
        paddedX = numpy.zeros((PADDED_WIDTH,PADDED_HEIGHT))
        paddedX[:X.shape[0],:X.shape[1]] = X[:SPEC_WIDTH,:SPEC_HEIGHT]
        
        #print(numpy.shape(paddedX))
        #nnti = paddedX[0:84, 0:126]
        #print(nnti)
        #print(numpy.shape(nnti))
        
        
        #if(file_count==56):
          #hola = librosa.griffinlim_cqt(X[:SPEC_WIDTH,:SPEC_HEIGHT],hop_length=512) #so we know that the inverse cqt actually works
          #hola = librosa.feature.inverse.mel_to_audio(M = X, sr=sr)
          #scipy.io.wavfile.write("audio_test.wav",16000,hola)
        #hola = librosa.core.griffinlim_cqt(nnti,sr=16000)#,hop_length=128,dtype = numpy.float64,bins_per_octave = int(36))
        
        #####################################
        ###################################saves as numpy binary
        filenameseg = output_dir+'/'+filename #output_dir/filename/.npy
        filenameseg = filenameseg+".npy"
        save(filenameseg,paddedX)
        
        
        file_count = file_count + 1
        
        
        

ConvertFolderToSpectrograms("audio/TrainA","output/TrainA")
ConvertFolderToSpectrograms("audio/TrainB","output/TrainB")
ConvertFolderToSpectrograms("audio/TestA","output/TestA")
ConvertFolderToSpectrograms("audio/TestB","output/TestB")
