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

nummffcs = 13
numT = 126
MFCCtrainA = numpy.zeros((139,nummffcs,numT))
MFCCtrainB = numpy.zeros((160,nummffcs,numT))
MFCCgen = numpy.zeros((1,nummffcs,numT))

def ConvertFolderToMFCC(input_dir,output_dir,mfccs,num):
    file_count = 0
    for filename in os.listdir(input_dir):
        if(file_count<num):
            x, sr = librosa.load(input_dir+'/'+filename, sr=None)
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=nummffcs)
      #      mfcc=numpy.mean(mfcc,axis=1)
            #print(mfcc)
            mfccs[file_count]=mfcc
            file_count = file_count + 1


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = numpy.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = numpy.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = numpy.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix



ConvertFolderToMFCC("audio/TrainA","output/EvalTrainB",MFCCtrainA,139)
ConvertFolderToMFCC("audio/TrainB","output/EvalTrainB",MFCCtrainB,160)
ConvertFolderToMFCC("audio/GeneratedB","output/EvalGeneratedB",MFCCgen,1)

def average(mfccarray):
    sum = 0
    count = 0
    for mfcc in mfccarray:
        sum = sum + mfcc
        count = count + 1
    return sum/count

from scipy import spatial
from scipy.spatial import minkowski_distance


def compareMFCC(mfccarray1,mfccarray2):
    count = 0
    sum = 0
    for x in mfccarray1:
        diff = x - mfccarray2[count]
    return count

def dtw_average(mfccarray1,mfccarray2):
    sum = 0
    count = 0
    for mfcc1 in mfccarray1:
        for mfcc2 in mfccarray2:
            for i in range(1):
            # cos_sim = dot(mfcc1[i], mfcc2[i])/(norm(mfcc1[i])*norm(mfcc2[i]))
                distance = spatial.distance.cosine(mfcc1[i],mfcc2[i])
            #    distance = euclidean(mfcc1[i],mfcc2[i])
            #    distance,path = fastdtw(mfcc1[i],mfcc2[i])
                sum = sum + distance
                count = count + 1
    return sum/count


#175.80532256029116
#182.74961202042618 new


#145.07586722797888 old
#142.54542347796558



#avgA = average(MFCCtrainA)
#avgB = average(MFCCtrainB)
#avgG = average(MFCCgen)

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

print(dtw_average(MFCCtrainA,MFCCgen))
print(dtw_average(MFCCtrainB,MFCCgen))


#dtwG_A = dtw(avgG,avgA)[-1][-1]
#dtwG_B = dtw(avgG,avgB)[-1][-1]

#print(dtwG_A)
#print(dtwG_B)

#print(MFCCtrain)
#print(MFCCgen)








#import matplotlib.pyplot as plt

#from sklearn import datasets
#from sklearn.decomposition import PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#X_a = MFCCtrainA.reshape((139,nummffcs))
#X_b = MFCCtrainB.reshape((139,nummffcs))
#X_gen = MFCCgen.reshape((10,nummffcs))
#y_a = numpy.full((139),0)
#y_b = numpy.full((139),1)
#y_gen = numpy.full((10),2)

#X = numpy.concatenate((X_a,X_b),axis=0)
#y = numpy.concatenate((y_a,y_b),axis=0)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)


##X = numpy.concatenate((X,X_gen),axis=0)
##y = numpy.concatenate((y,y_gen),axis=0)
##print(X.shape)


#iris = datasets.load_iris()
#target_names = ["string","flute"]



#pca = PCA(n_components=2)
#X_r = pca.fit(X).transform(X)

#import pandas as pd

#print(pd.DataFrame(pca.components_,index = ['PC-1','PC-2']))


#from sklearn.ensemble import RandomForestClassifier

#classifier = RandomForestClassifier(max_depth=2, random_state=0)
#classifier.fit(X, y)

## Predicting the Test set results
#y_pred = classifier.predict(X_gen)

##transform and add to existing space

#print(X_r.shape)


#lda = LinearDiscriminantAnalysis(n_components=1)
#X_r2 = lda.fit(X_r, y).transform(X_r)



##print(lda.predict(X_gen))

##print(lda.predict(X_gen))


## Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s'
      #% str(pca.explained_variance_ratio_))

#plt.figure()
#colors = ['navy', 'turquoise']#, 'darkorange']
#lw = 2

#for color, i, target_name in zip(colors, [0, 1], target_names):
    #plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                #label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('pca')


##pure euclid + cosine similarity

#plt.figure()
#for color, i, target_name in zip(colors, [0, 1], target_names):
    #plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha=.8, color=color,
                #label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('lda')





#plt.show()


