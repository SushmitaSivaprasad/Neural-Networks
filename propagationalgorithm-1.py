import numpy as np
import matplotlib as plot
X = np.array(([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]), dtype=float)
y = np.array(([0],[1],[1],[0],[1],[0],[0],[1],[1],[0],[0],[1],[0],[1],[1],[0]))
#print X.shape, y.shape
class Neural_Network(object):
    def __init__(self,n):        
        #Size of the arrays are decided 
        
        self.numberofinputs = 4
        self.numberofoutputs = 1
        self.hiddenlayernodes = 4
        self.eta = n
        self.temp1=0
        self.temp2=0
        self.correct=0
        np.random.seed(2000)
        #random selection of weights between 0 and 1 
        
        self.W1 = 2*np.random.random((self.numberofinputs, self.hiddenlayernodes))-1
        self.W2 = 2*np.random.random((self.hiddenlayernodes, self.numberofoutputs))-1
        
        #defining the size of the updated weights 
        
        self.dJdW1 = np.random.random((self.numberofinputs, self.hiddenlayernodes))
        self.dJdW2 = np.random.random((self.hiddenlayernodes, self.numberofoutputs))
        
        # forward propagation algorithm
        
    def forward(self,W1,W2,X):
        self.v2 = np.dot(X, W1)
        self.phi2 = self.sigmoid(self.v2)
        self.v3 = np.dot(self.phi2, W2)
        ypred = self.sigmoid(self.v3) 
        self.correct=0
        return ypred
        
        # defining the sigmoid fucntion 
    def sigmoid(self, z):
       
        return 1/(1+np.exp(-z))
        
        
    def cost(self, X, y,W1,W2):
        
        self.ypred = self.forward(W1,W2,X)
        J = 0.5*sum((y-self.ypred)**2)
        return J
        
    def backward(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.ypred = self.forward(self.W1,self.W2,X)
        
        delta3 = np.multiply((y-self.ypred), self.sigmoidderv(self.v3))
        self.dJdW2 = np.dot(self.phi2.T, delta3)
      
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidderv(self.v2)
        self.dJdW1 = np.dot(X.T, delta2) 
        
        #derivative of sigmoid function
    def sigmoidderv(self,z):
        
        return np.exp(-z)/((1+np.exp(-z))**2)
    
        return self.dJdW1, self.dJdW2
    
    def updateweights(self):
    
        self.W1+=self.eta*self.dJdW1
        self.W2+=self.eta*self.dJdW2


        self.temp1 = 0.9*self.temp1+ n*self.dJdW1
        self.W1 += self.temp1
        self.temp2  = 0.9 *self.temp2 +n*self.dJdW2
        self.W2 += self.temp2
          
        
    def num_correct_pred(self):
 
          
      return sum(abs(self.ypred-y <=0.05))[0]
    
      
     

eta_list=[0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]


for n in eta_list:
  NN = Neural_Network(n) 

  for i in range(100000000):
      NN.forward(NN.W1,NN.W2,X)
      NN.backward(X,y)
      NN.updateweights()
      if NN.num_correct_pred()>15:
        print ("For learning rate ",n," Convergence at Epoch ",i,NN.num_correct_pred())
        break
      if i%50000==0:
        print (i,NN.num_correct_pred())

      