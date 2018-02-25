# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:31:23 2018

@author: HIGH TECH
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from spectrum import *
#Heart rate calculation and record plotting



def parseFile(filename):
    file = open("D:\\Faculty of engineering\\Semester 10\\GP2\\ECG_data\\Text_records\\Normalized\\Rec_" + filename + ".txt")
    split1 = file.read().split('\n[')
    data = []
    labels = []
    for i in split1:
        temp = i.split('\t')
        data.append(cleanString(temp[0]))
        labels.append(temp[1].replace("\n", ""))
    return data,labels
    
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




def cleanString(st):
    st = st.replace("[", "")
    st = st.replace("]", "")
    st = st.replace("\n", " ")
    
    st = " ".join(st.split())
    listofvalues = st.split(' ')
    results = list(map(float, listofvalues))
    return results

def convertLabels(labels):
    newlabels = np.empty(len(labels), dtype=float);
    for i in range(len(newlabels)):
        #print(i)
        newlabels[i] = {
                'N': 0,
                'V': 1,
                'Q': 2,
                '+': 3,
                '~': 4,
                '|': 5,
                'A': 6
                }[labels[i]]
    return newlabels

"""
file = open("D:\\Faculty of engineering\\Semester 10\\GP2\\ECG_data\\Text_records\\Normalized\\Rec_103.txt")
split1 = file.read().split('\n[')
data = []
labels = []
for i in split1:
    temp = i.split('\t')
    data.append(cleanString(temp[0]))
    labels.append(temp[1].replace("\n", ""))
"""
data = []
labels = []
for i in range(100,103):
    print("Procecssing file " + str(i))
    if(i%10 == 0):
        continue
    d,l = parseFile(str(i))
    data.append(d)
    labels.append(l)

data = [item for items in data for item in items]
labels = [item for items in labels for item in items]
#data = flatten(data)
#labels = flatten(labels)
#print("Data length = " + str(len(data)))
#print("Label length = " + str(len(labels)))
#min_max_scaler = preprocessing.MinMaxScaler()
#data = min_max_scaler.fit_transform(data)

#s = preprocessing.StandardScaler()
#data = s.fit_transform(data)
#print(labels[2273])
#print(convertLabels(labels))

yule4data = []
yule8data = []
yule20data = []

burg4data = []
burg8data = []
burg20data = []

print("=========PROCESSING YULE 4==========\n")
for i in range(len(data)):
    ARyule4, Pyule4, kyule4 = aryule(data[i],4)
    yule4data.append(ARyule4)

print("KNN ACCURACY : " + str(classifyKNN(yule4data,labels)) +"%")
#print("SVM ACCURACY : " + str(classifySVM(yule4data,labels)))
classifySVM(yule4data,labels)

print("\n=========PROCESSING YULE 8==========\n")
for i in range(len(data)):
    ARyule8, Pyule8, kyule8 = aryule(data[i],8)
    yule8data.append(ARyule8)
print("KNN ACCURACY : " + str(classifyKNN(yule8data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule8data,labels)))

print("\n=========PROCESSING YULE 16==========\n")    
for i in range(len(data)):
    ARyule20, Pyule20, kyule20 = aryule(data[i],20)
    yule20data.append(ARyule20)

print("KNN ACCURACY : " + str(classifyKNN(yule20data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule20data,labels)))


print("=========PROCESSING BURG 4==========\n")
for i in range(len(data)):
    ARburg4, Pburg4, kburg4 = arburg(data[i],4)
    burg4data.append(ARburg4)

print("KNN ACCURACY : " + str(classifyKNN(burg4data,labels))+"%")
#print("SVM ACCURACY : " + str(classifySVM(yule4data,labels)))
#classifySVM(4data,labels)

print("\n=========PROCESSING BURG 8==========\n")
for i in range(len(data)):
    ARburg8, Pburg8, kburg8 = arburg(data[i],6)
    burg8data.append(ARburg8)
print("KNN ACCURACY : " + str(classifyKNN(burg8data,labels)))


"""
for i in split1:
    temp = i.split('       ')
    print(temp)
"""