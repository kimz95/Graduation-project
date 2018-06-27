# -*- coding: utf-8 -*-
"""

@author: HIGH TECH
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from spectrum import *
#Heart rate calculation and record plotting

def getKValueOfFile(filename,filenames,kvalues):
    i = filenames.index(filename)
    return kvalues[i]

def getKValues():
    file = open("D:\\Faculty of engineering\\Semester 10\\GP2\\ECG_data\\Text_records\\Normalized\\kvalues.txt")
    split1 = file.read().split('\n')
    filename = []
    kvalues = []
    for i in split1:
        temp = i.split(' ')
        filename.append(int(temp[0][4:]))
        kvalues.append(int(temp[1].replace("\n", "")))
    #print (filename)
    #print(kvalues)
    return filename,kvalues

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
    
def classifyKNN(data,labels,k):
    limit = int(0.70* len(data))
    learndata = data[0:limit]
    learnlabels = labels[0:limit]
    testdata = data[limit:]
    testlabels = labels[limit:]       
        

    
    neigh = KNeighborsClassifier(n_neighbors=k)
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

def bestyule(yule4,yule8,yule20):
    Max = 0
    if(yule4>yule8 and yule4>yule20):
        print("Best is YULE4 with accuracy " + str(yule4))
    elif(yule8>yule4 and yule8>yule20):
        print("Best is YULE8 with accuracy " + str(yule8))
    elif(yule20>yule4 and yule20>yule8):
        print("Best is YULE20 with accuracy " + str(yule20))
        
    
    
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
filenames = []
kvalues = []
filenames, kvalues = getKValues()
for i in range(203,204):
    print("Procecssing file " + str(i))
    if(i%10 == 0):
        continue
    k = getKValueOfFile(i,filenames,kvalues)
    print("K VALUE OF " + str(i) + " IS " + str(k))
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


print("=========PROCESSING YULE 4==========\n")
for i in range(len(data)):
    ARyule4, Pyule4, kyule4 = aryule(data[i],4)
    yule4data.append(ARyule4)

yule4res = classifyKNN(yule4data,labels,k)
print("KNN ACCURACY : " + str(yule4res) +"%")
print("SVM ACCURACY : " + str(classifySVM(yule4data,labels)))
classifySVM(yule4data,labels)

print("\n=========PROCESSING YULE 8==========\n")
for i in range(len(data)):
    ARyule8, Pyule8, kyule8 = aryule(data[i],8)
    yule8data.append(ARyule8)
yule8res = classifyKNN(yule8data,labels,k)
print("KNN ACCURACY : " + str(yule8res)+"%")
print("SVM ACCURACY : " + str(classifySVM(yule8data,labels)))

print("\n=========PROCESSING YULE 20==========\n")    
for i in range(len(data)):
    ARyule20, Pyule20, kyule20 = aryule(data[i],20)
    yule20data.append(ARyule20)
yule20res = classifyKNN(yule20data,labels,k)
print("KNN ACCURACY : " + str(yule20res)+"%")
print("SVM ACCURACY : " + str(classifySVM(yule20data,labels)))
bestyule(yule4res,yule8res,yule20res)

"""
for i in split1:
    temp = i.split('       ')
    print(temp)
"""