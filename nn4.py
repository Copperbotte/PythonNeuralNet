
import numpy as np
import random as rand

#neural net generating functions
def _makeNNet(n1, n2, func):
    #from n1+1 to n2
    return np.array([[func(x,y) for x in range(n1+1)] for y in range(n2)])

def makeZeroNNet(n1, n2):
    def zeroFunc(x,y):
        return 0.0
    return _makeNNet(n1, n2, zeroFunc)

def makeIdentityNNet(n1, n2):
    def identityFunc(x,y):
        return 1.0 if x == y else 0.0
    return _makeNNet(n1, n2, identityFunc)

def makeRandNNet(n1, n2):
    def randFunc(x,y):
        return rand.random()
    return _makeNNet(n1, n2, randFunc)

def buildNNet(func, nodes):
    return [func(a,b) for a,b in zip(nodes[:-1], nodes[1:])]

#sigmoids
def sigTanh(x):
    return np.tanh(x)

def dsigTanh(sig):
    return 1.0 - sig**2

getdSig = {sigTanh: dsigTanh}

#States is the same length as the first node in Nodes.
#Sigmoid is performed before the transform rather than after,
#   to accomodate for non-bounded neural net outputs.
def forwardprop(nnet, states):
    res = [states]
    sigs = []
    for W in nnet:
        sig = [1.0] + list(map(sigmoid, res[-1]))#sigmoid input
        N1 = np.array([sig]).transpose()        #build vector
        N2 = W @ N1                             #mul
        res.append(N2.transpose().tolist()[0])  #un-build vector
        sigs.append(sig)
    return res, sigs

def backprop(nnet, nodes, states, expect):
    #forwardprop
    res, sigs = forwardprop(nnet, states)
    
    #backprop
    delta = buildNNet(makeZeroNNet, Nodes)

    dN = [r-e for r,e in zip(res[-1], expect)]  #delta node
    error = sum([e**2 / 4.0 for e in dN])       #numerical error
    dNode = np.array([dN]).transpose()          #delta node vector

    for N in range(len(nnet)):
        n = len(nnet) - N - 1 #reverse order
        #find weights
        #deltaNNet is the unique combo of dN2 * sigma(N1)
        # ----- I feel like this can be simplified to a matrix mul -----
        for y in range(len(delta[n])):
            for x in range(len(delta[n][0])):
                delta[n][y][x] += sigs[n][x] * dNode[y][0]

        #backprop to next node
        dNode = nnet[n].transpose() @ dNode #weight derivative
        dNode = dNode[1:]                   #strip bias
        #sigmoid derivative
        dNode *= np.array([list(map(getdSig[sigmoid], sigs[n][1:]))]).transpose()
    
    return res, error, delta

#Nodes includes start and end nodes
Nodes = [2, 2, 1]

#Define sigmoid used
sigmoid = sigTanh

#NNet includes a hidden bias node from its source as the 1st element.
#Each matrix transforms between two Nodes elements.
NNet = buildNNet(makeRandNNet, Nodes)

print("Nodes")
print(Nodes)
for n in NNet:
    print(n)
    print()

for times in range(100):
    Result, Error, Delta = backprop(NNet, Nodes, [1,1], [1])
    print(times, Result[-1][0], Error)
    for n in range(len(NNet)):
        NNet[n] -= 0.5 * Delta[n]


    
