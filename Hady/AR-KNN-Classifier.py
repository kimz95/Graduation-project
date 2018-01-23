import matplotlib.pyplot as plt
import numpy as np
import math
import pywt
import wfdb
from spectrum import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Heart rate calculation and record plotting

def classifySVM(data,labels):
    limit = int(0.75* len(data))
    learndata = data[0:limit]
    learnlabels = labels[0:limit]
    testdata = data[limit:]
    testlabels = labels[limit:]       
        

    
    SVM= SVC(kernel ='poly')
    SVM.fit(learndata, learnlabels) 
    #print(len(learndata))
    #print(len(learndata[0]))    
    predicted = SVM.predict(testdata)
    return(accuracy_score(testlabels,predicted)*100)
    
def classifyKNN(data,labels):
    limit = int(0.75* len(data))
    learndata = data[0:limit]
    learnlabels = labels[0:limit]
    testdata = data[limit:]
    testlabels = labels[limit:]       
        

    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(learndata, learnlabels) 
    #print(len(learndata))
    #print(len(learndata[0]))    
    predicted = neigh.predict(testdata)
    return(accuracy_score(testlabels,predicted)*100)
    
    #res = 0
    #for i in range(len(testlabels)):
    #    if(predicted[i] == testlabels[i]):
    #        res+=1 
    #print("Accuracy is " + str(res) + " out of " + str(len(testlabels)) + "  :  " + str((res/len(testlabels))*100) + "%")

def peaks(x, peak_indices, fs, title, figsize=(20, 10), saveto=None):
    
    hrs = wfdb.processing.compute_hr(siglen=x.shape[0], peak_indices=peak_indices, fs=fs)
    return hrs

#Single beats extraction for record
def extract_beats(record, R_indices, peaks_hr, freq):
    
    flat_record = [item for sublist in record for item in sublist]
    ecg_beats = [];
    
#Extract beats of different lengths depending on heart rate
    for i in range(0, len(R_indices)):
        hr = peaks_hr[R_indices[i]].item();
        if(math.isnan(hr)):
            hr = 70.0;
        samples_per_beat = int(freq*(60.0/hr))
        start = R_indices[i]-samples_per_beat/2;
        if(start<0):
            start = 0;
        end = start + samples_per_beat;
        ecg_beats.append(np.array(flat_record[int(start):int(end)]));
    
#Resample beats to get fixed input size for classification
    ecg_nparray = np.empty((len(ecg_beats),128))
    for i in range(0,len(ecg_beats)):    
        ecg_beats[i], _ = wfdb.processing.resample_sig(x=ecg_beats[i],fs=len(ecg_beats[i]), fs_target=128);
        ecg_nparray[i] = ecg_beats[i];
            
    return ecg_nparray;

def read_record(rec, t0=0, tf=300000):
# Load the wfdb record and the physical samples
    annotation = wfdb.rdann('C:\\GradProj\\olddataset\\'+rec, 'atr', sampfrom=t0, sampto=tf,summarize_labels=True)
    print(annotation.symbol)
    record = wfdb.rdsamp('C:\\GradProj\\olddataset\\'+rec, sampfrom=t0, sampto=tf, channels=[0])
    freq = record.fs
    sig = record.p_signals
    sig = wfdb.processing.normalize(x=sig, lb=0.0, ub=1.0)

    for idx, val in enumerate(sig):    
        record.p_signals[idx,0] = val

    peak_indices = annotation.sample;
    peaks_hr = peaks(x=record.p_signals, peak_indices=peak_indices, fs=record.fs, title="GQRS peaks on record "+rec);
    ecg_beats = extract_beats(record.p_signals, peak_indices, peaks_hr,freq);
    
    return ecg_beats, annotation;
    
t0=0;
tf=600000;
beats = [];
beats, annotation = read_record("105", t0, tf)

#fig, ax_left = plt.subplots(figsize=(20,10))
#ax_left.plot(beats[0], color='#3979f0', label='Signal')
#ax_left.set_xlabel('Time (ms)')
#ax_left.set_ylabel('ECG (mV)', color='#3979f0')
#ax_left.tick_params('y', colors='#3979f0')
#labels preparation
X = beats;
N = len(beats);
labels = np.empty(N, dtype=int);
for i in range(0,N):
    labels[i] = {
        'N': 0,
        'V': 1,
        'Q': 2,
        '+': 3,
        '~': 4,
        '|': 5,
        }[annotation.symbol[i]]

yule4data = []
yule8data = []
yule20data = []

burg4data = []
burg8data = []
burg20data = []

print("=========PROCESSING YULE 4==========\n")
for i in range(len(beats)):
    ARyule4, Pyule4, kyule4 = aryule(X[i],4)
    yule4data.append(ARyule4)

print("KNN ACCURACY : " + str(classifyKNN(yule4data,labels)) +"%")
#print("SVM ACCURACY : " + str(classifySVM(yule4data,labels)))
classifySVM(yule4data,labels)

print("\n=========PROCESSING YULE 8==========\n")
for i in range(len(beats)):
    ARyule8, Pyule8, kyule8 = aryule(X[i],8)
    yule8data.append(ARyule8)
print("KNN ACCURACY : " + str(classifyKNN(yule8data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule8data,labels)))

print("\n=========PROCESSING YULE 16==========\n")    
for i in range(len(beats)):
    ARyule20, Pyule20, kyule20 = aryule(X[i],20)
    yule20data.append(ARyule20)

print("KNN ACCURACY : " + str(classifyKNN(yule20data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule20data,labels)))


print("=========PROCESSING BURG 4==========\n")
for i in range(len(beats)):
    ARburg4, Pburg4, kburg4 = arburg(X[i],4)
    burg4data.append(ARburg4)

print("KNN ACCURACY : " + str(classifyKNN(burg4data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule4data,labels)))
#classifySVM(4data,labels)

print("\n=========PROCESSING BURG 8==========\n")
for i in range(len(beats)):
    ARburg8, Pburg8, kburg8 = arburg(X[i],8)
    burg8data.append(ARburg8)
print("KNN ACCURACY : " + str(classifyKNN(burg8data,labels)))
#print("SVM ACCURACY : " + str(classifySVM(yule8data,labels)))

"""
print("\n=========PROCESSING BURG 16==========\n")    
for i in range(len(beats)):
    ARburg20, Pburg20, kburg20 = arburg(X[i],16)
    burg20data.append(ARburg20)

print("KNN ACCURACY : " + str(classifyKNN(burg20data,labels)))
#print("SVM ACCURACY : " + str(classifySVM(yule20data,labels)))
"""
#HIGH ORDER WITH BURG CAUSES NEGATIVE NUMBERS, MAKING THE CLASSIFIER NOT WORK

#ARburg8, Pburg8, kburg8 = arburg(X[0],8)
#ARburg16, Pburg16, kburg16 = arburg(X[0],16)

#ARyule4, Pyule4, kyule4 = aryule(X[0],4)
#ARyule8, Pyule8, kyule8 = aryule(X[0],8)
#ARyule16, Pyule16, kyule16 = aryule(X[0],16)

#print(len(ARyule16))


