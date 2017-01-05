import numpy as np
from matplotlib import pyplot as plt
import time
import struct

#Function generators
def XOR_GEN():
    I = np.random.rand(1,2)
    tI = np.round(I).astype(int)
    O = tI[0][0]^tI[0][1]
    O = np.asarray(O)
    return I,O
def MULT_GEN():
    I = np.random.rand(1,2)
    O = I[0][0]*I[0][1]
    O = np.asarray(O)
    return I,O
def MOD_GEN():
    I = np.random.rand(1,2)
    O = np.mod(I[0][0],I[0][1])
    O = np.asarray(O)
    return I,O
def AVG_GEN():
    I = np.random.rand(1,2)
    O = (I[0][0]+I[0][1])/2
    O = np.asarray(O)
    return I,O
def SWAP_GEN():
    I = np.random.rand(1,2)
    O = (I[0][1],I[0][0])
    O = np.asarray(O)
    return I,O

def GEN():
    return XOR_GEN()


#TRANSFER FUNCTION
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def sigmoidPrime(x):
    s = sigmoid(x)
    return s*(1.0-s)


#NETWORK
class Layer:
    def __init__(self,n):
        pass
class Net:
    def __init__(self,t):
        self.W = [] #weight matrices
        self.L = [] #Layers
        self.B = [] #Biases
        self.length = len(t)
        for l in t:
            self.L.append(Layer(l))
        for i in range(1,self.length):
            self.W.append(np.random.randn(t[i-1],t[i]))
            self.B.append(np.random.randn(1,t[i]))
        print self.W
    def FF(self,X):
        self.L[0].O = X
        for i in range(1,self.length):
            self.L[i].I = np.dot(self.L[i-1].O,self.W[i-1]) + self.B[i-1]
            self.L[i].O = sigmoid(self.L[i].I)
        return self.L[-1].O
    def BP(self,X,Y):
        G = Y - self.FF(X) #ERROR Gradient : differentiated result of 0.5*(Y-X)^2
        self.L[-1].G = G# * sigmoidPrime(self.L[-1].O)
        #calc Gradient
        for i in reversed(range(1,self.length-1)):
            self.L[i].G = np.dot(self.L[i+1].G,self.W[i].T) * sigmoidPrime(self.L[i].I)
        for i in range(1,self.length):
            self.W[i-1] += 0.6 * np.dot(self.L[i-1].O.T, self.L[i].G) 
            self.B[i-1] += 0.6 * self.L[i].G
        return 0.5 * np.sum(G*G)

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def main():
    t = [2,4,1]
    net = Net(t)
    G = []
    start = time.time()
    
    for i in range(10000):
        I,O = GEN()
        net.BP(I,O)
        G.append(net.BP(I,O))
    I,O = GEN()

    end = time.time()
    print("TOOK {} SECONDS".format(end-start))
    print("INPUT : ", I)
    print("TARGET : ", O)
    print("OUT : ", net.FF(I))

    np.set_printoptions(formatter={'float':float_to_hex})
    #np.array2string(x, formatter={'int':lambda x: hex(x)})
    #np.savetxt('W.txt', net.W[0], fmt="%X")

    print np.ndarray.flatten(net.W[0])
    print np.ndarray.flatten(net.W[1])

    print np.ndarray.flatten(net.B[0])
    print np.ndarray.flatten(net.B[1])

    plt.plot(G)
    plt.show()


if __name__ == "__main__":
    main()
