import csv
import random
import math
import operator
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

        for row in lines:
            print ', '.join(row)


def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)


def getNeightbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testset, preditions):
    correct=0
    for x in range(len(testset)):
        if testset[x][-1] == preditions[x]:
            correct +=1
    return (correct/float(len(testset))) *100.0

def main():
    trainingSet=[]
    testSet=[]
    split=0.67
    loadDataset('iris.data',split,trainingSet,testSet)
    predicitons=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeightbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predicitons.append(result)
        print('>predictions = ' + repr(result) + ', actual = ' + testSet[x][-1])
    accuray = getAccuracy(testSet,predicitons)
    print('Accuracy: ' + repr(accuray) +'%')
    print 'Train : ' + repr(len(trainingSet))
    print 'Test: ' + repr(len(testSet))

main()