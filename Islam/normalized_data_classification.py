import numpy as np
from sklearn.svm import *
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score


def classifyAllSVM(data,labels):
    limit = int(0.75* len(data))
    learndata = data[0:limit]
    learnlabels = labels[0:limit]
    testdata = data[limit:]
    testlabels = labels[limit:] 
    print("CLASSIFYING : 75\% LEARNING, 25\% TESTING")
    print("------------    Linear SVC   -----------\n")
    classifier = LinearSVC()
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels,predicted)*100))

    print("------------    SVC(linear)   -----------\n")

    classifier = SVC(kernel='linear')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels,predicted)*100))

    print("------------    SVC(poly)   -----------\n")

    classifier = SVC(kernel='poly')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels,predicted)*100))

    print("------------    SVC(rbf)   -----------\n")

    classifier = SVC(kernel='rbf')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels,predicted)*100))

    print("------------    SVC(sigmoid)   -----------\n")

    classifier = SVC(kernel='sigmoid')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels,predicted)*100))

    print("------------    Linear SVR   -----------\n")

    classifier = LinearSVR()
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))


    print("------------    SVR(linear)   -----------\n")

    classifier = SVR(kernel='linear')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))

    print("------------    SVR(poly)   -----------\n")

    classifier = SVR(kernel='poly')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))

    print("------------    SVR(rbf)   -----------\n")

    classifier = SVR(kernel='rbf')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))

    print("------------    SVR(sigmoid)   -----------\n")

    classifier = SVR(kernel='sigmoid')
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))
    """
    print("------------    l1_min_c   -----------\n")

    classifier = l1_min_c()
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))
    """
    print("------------    NuSVC   -----------\n")

    classifier = NuSVC(nu=0.0009)
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))
    
    print("------------    NuSVR   -----------\n")

    classifier = NuSVR(0.5)
    classifier.fit(learndata,learnlabels)
    predicted = classifier.predict(testdata)
    print("Accuracy = " + str(accuracy_score(testlabels, predicted.round(), normalize=False)/len(testdata)*100))
    


def parseFile(filename):
    file = open("C:\\ECG_data\\Text_records\\Normalized\\Rec_" + filename + ".txt")
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
                'A': 6,
                '/': 7,
                'f': 8
                }[labels[i]]
    return newlabels

def singleFilePrediction(filename):
    print("Procecssing file " + filename)
    data,labels = parseFile(filename)
    labels = convertLabels(labels)
    classifyAllSVM(data,labels)

def multiFilePrediction(r1,r2):
    data = []
    labels = []
    for i in range(r1,r2):
        print("Procecssing file " + str(i))
        if(i%10 == 0):
            continue
        d,l = parseFile(str(i))
        data.append(d)
        labels.append(l)
    data = [item for items in data for item in items]
    labels = [item for items in labels for item in items]
    labels = convertLabels(labels)
    classifyAllSVM(data,labels)


#multiFilePrediction(100,103)
singleFilePrediction("102")