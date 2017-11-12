from numpy import *
import operator

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
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    fr=open(filename)
    index=0
    for line in fr.readlines():
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
    DatingDataMat,datingLabels = file2matrix('')
    normMat,ranges,minVal = autoNorm(DatingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*h0Ratio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifirResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        if(classifirResult != datingLabels[i]):
            errorCount +=1
    return errorCount


def main():
    group,labels=createDataSet()
    inX=[0,0]
    autoData,range,minValue=autoNorm(group)
    result=classify0(inX,autoData,labels,3)
    print(result)
    print(autoData)

main()



