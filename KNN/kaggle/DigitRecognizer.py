from numpy import *
import operator
import matplotlib
from os import listdir
import matplotlib.pyplot as plt

import csv


def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances =sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines=len(arrayLines)
    returnMat=zeros((numberOfLines,784))
    classLabelVector=[]
    index=0
    for line in arrayLines:
        if index != 0:
            line=line.strip()
            listFromLine=line.split(',')
            returnMat[index,:]=listFromLine[1:]
            classLabelVector.append(int(listFromLine[0]))
        index +=1
    return nomalizing(toInt(returnMat)),classLabelVector


def saveResult(result):
    with open('result.csv','wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j]=int(array[i,j])
    return newArray


def datingClassTest():
    h0Ratio = 0.50
    DatingDataMat,datingLabels = file2matrix('/Users/tangang/SourceCode/kaggledata/DigitRecognizer/train.csv')
    #normMat,ranges,minVal = autoNorm(DatingDataMat)
    #m=DatingDataMat.shape[0]
    m=100
    numTestVecs=int(50)
    errorCount=0.0
    for i in range(numTestVecs):
        classifirResult=classify0(DatingDataMat[i,:],DatingDataMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        if(classifirResult != datingLabels[i]):
            errorCount +=1
    print '\n the total number of errors is : %d' %errorCount
    print '\n the total error rate is:%f' % (errorCount/float(numTestVecs))
    return errorCount

def loadTestData():
    l=[]
    with open('/Users/tangang/SourceCode/kaggledata/DigitRecognizer/test.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
            #28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))

def handwritingClassTest():

    DatingDataMat,trainLabel = file2matrix('/Users/tangang/SourceCode/kaggledata/DigitRecognizer/train.csv')

    testData=loadTestData()

    m,n=shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):
        classifierResult = classify0(testData[i], DatingDataMat, trainLabel, 5)
        resultList.append(classifierResult)
    saveResult(resultList)




def main():
    # group,labels=createDataSet()
    #inX=[0,0]
    #autoData,range,minValue=autoNorm(group)
    #result=classify0(inX,autoData,labels,3)
    #print(result)
    #print(autoData)
    #datingClassTest()
    #drawplot()
    #handwritingClassTest()
    #returnMat,classLabelVector = file2matrix('/Users/tangang/SourceCode/kaggledata/DigitRecognizer/train.csv')
    handwritingClassTest()
    #print(classLabelVector)

main()



