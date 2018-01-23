import wfdb
import matplotlib.pyplot as plt
import numpy as np
import math
import pywt
from sklearn.svm import SVC
#Heart rate calculation and record plotting
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
    ecg_nparray = np.empty((len(ecg_beats),600))
    for i in range(0,len(ecg_beats)):    
        ecg_beats[i], _ = wfdb.processing.resample_sig(x=ecg_beats[i],fs=len(ecg_beats[i]), fs_target=600);
        ecg_nparray[i] = ecg_beats[i];
            
    return ecg_nparray;

def read_record(rec, t0=0, tf=300000):
# Load the wfdb record and the physical samples
    record = wfdb.rdsamp('C:\\Users\\Islam\\Desktop\\Graduation-project-master\\Kareem\\dataset\\'+rec, sampfrom=t0, sampto=tf, channels=[0])
    annotation = wfdb.rdann('C:\\Users\\Islam\\Desktop\\Graduation-project-master\\Kareem\\dataset\\'+rec, 'atr', sampfrom=t0, sampto=tf,summarize_labels=True)
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
#wavlet transfrom 
wavelets=[]
for i in range(len(beats)):
    (ca, cd) = pywt.dwt(beats[i],'haar')
    cat = pywt.threshold(ca,np.std(ca)/12)
    cdt = pywt.threshold(cd,np.std(cd)/12)
    ts_rec = pywt.idwt(cat, cdt, 'haar')
    wavelets.append(ts_rec)
#
fig, ax_left = plt.subplots(figsize=(20,10))
ax_left.plot(beats[0], color='#3979f0', label='Signal')
ax_left.plot(ts_rec, color='#000000', label='Decomposed')
ax_left.set_xlabel('Time (ms)')
ax_left.set_ylabel('ECG (mV)', color='#3979f0')
ax_left.tick_params('y', colors='#3979f0')
#labels preparation
X = beats;
N = len(beats);
Y = np.empty(N, dtype=int);
for i in range(0,N):
    Y[i] = {
        'N': 0,
        'V': 1,
        'Q': 2,
        '+': 3,
        '~': 4,
        '|': 5,
        }[annotation.symbol[i]]
#training (75%) and (25%)testing data 
limit = int(0.75* len(wavelets))
learndata = wavelets[0:limit]
learnlabels = Y[0:limit]
testdata = wavelets[limit:]
testlabels = Y[limit:]       
        
clf = SVC()
clf.fit(learndata, learnlabels)
print(len(learndata))
print(len(learndata[0]))    
predicted = clf.predict(testdata)
res = 0
for i in range(len(testlabels)):
    if(predicted[i] == testlabels[i]):
       res+=1 
print("Accuracy is " + str(res) + " out of " + str(len(testlabels)) + "  :  " + str((res/len(testlabels))*100) + "%")
# accuracy result is 94.3% on record 105 for example
