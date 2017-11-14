from numpy import *
import operator
import matplotlib
from os import listdir
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0,110],[1.0,100],[1,120],[0,105]])
    labels = ['A','A','B','B']
    return group,labels

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
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector




def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    range=maxVals-minVals
    m=dataSet.shape[0]
    normDataSet=zeros(shape(dataSet))
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(range,(m,1))
    return normDataSet,range,minVals

def datingClassTest():
    h0Ratio = 0.50
    DatingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVal = autoNorm(DatingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*h0Ratio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifirResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        if(classifirResult != datingLabels[i]):
            errorCount +=1
    return errorCount

def drawplot():
    DatingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DatingDataMat[:,1],DatingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect

def handwritingClassTest():
    hwLabels =[]
    trainnigFileList = listdir('trainingDigits')
    m = len(trainnigFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainnigFileList[i]
        fileStr= fileNameStr.split('.')[0]
        classNumStr=int((fileStr.split('_'))[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        if(classifierResult != classNumStr):
            errorCount += 1
    print '\n the total number of errors is : %d' %errorCount
    print '\n the total error rate is:%f' % (errorCount/float(mTest))




def main():
   # group,labels=createDataSet()
    #inX=[0,0]
    #autoData,range,minValue=autoNorm(group)
    #result=classify0(inX,autoData,labels,3)
    #print(result)
    #print(autoData)
   #datingClassTest()
   #drawplot()
   handwritingClassTest()

main()



