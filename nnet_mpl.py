import matplotlib.pyplot as plt
import numpy as np
import random as rand
import nn4


#Nodes includes start and end nodes
Nodes = [2, 5, 5, 1]

#Define sigmoid used
sigmoid = nn4.sigTanh

#NNet includes a hidden bias node from its source as the 1st element.
#Each matrix transforms between two Nodes elements.
NNet = nn4.buildNNet(nn4.makeRandNNet, Nodes)

def datafunc(x, y):
    return np.sin(x*2*np.pi) * np.sin(y*2*np.pi)

#make dataset
Dataset = []
for d in range(1000):
    x = rand.random() * 2 - 1
    y = rand.random() * 2 - 1
    c = datafunc(x ,y)
    Dataset.append([x,y,c])

def displayNNet(nnet, error):
    print(error)
    #xres and yres should be 369 x 368 to fit the window
    xres = 25
    yres = 25
    img = []
    for Y in range(yres):
        line = []
        y = -2*(Y/yres) + 1
        for X in range(xres):
            x = 2*(X/xres) - 1
            Result = nn4.forwardprop(nnet, [x, y], sigmoid)[0][-1][0]
            #clamp output, map to green and red
            c = np.tanh(Result)
            color = [0,c,0]
            if c < 0.0:
                color = [-c,0,0]
            color = [np.power(v, 1.0/2.2) for v in color]
            line.append(color)
        img.append(line)
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.clf()
    #plt.scatter([(d[0] + 1)*xres/2 for d in Dataset[:limit]], [(d[1] + 1)*yres/2 for d in Dataset[:limit]])
    plt.imshow(img)
    plt.draw()
    plt.pause(0.00001)

def mainLoop():
    #calculate delta
    global NNet, Nodes, Dataset, sigmoid
    rate = 0.5
    groupdelta = nn4.buildNNet(nn4.makeZeroNNet, Nodes)
    grouperror = 0
    for d in Dataset:
        result, error, delta = nn4.backprop(NNet, Nodes, d[:2], d[2:], sigmoid)
        for n in range(len(NNet)):
            groupdelta[n] += delta[n] / len(Dataset)
        grouperror += error / len(Dataset)
    for n in range(len(NNet)):
        NNet[n] -= rate * groupdelta[n]
    #display nnet
    displayNNet(NNet, grouperror)
    
loop = True
def on_close(evt):
    global loop
    print("Nodes")
    print(Nodes)
    for n in NNet:
        print(n)
        print()
    loop = False

plt.figure().canvas.mpl_connect('close_event', on_close)

while loop:
    mainLoop()
