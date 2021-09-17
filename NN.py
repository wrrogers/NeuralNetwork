import numpy as np

class NN (object):
    def __init__ (self, inp, lr = .001):
        self.inp = inp
        self.lr = lr
        self.names = []
        self.layer_count = 0
        self.synapses = []
        self.layers = []
        self.layersΔ = []
        
    def describeSyn(self):
        print("There are", len(self.synapses), "layers of synapses")
        for n, layer in enumerate(self.synapses):
            n += 1
            print("Layer", n, "-")
            print(layer)
            
    def describeLayers(self):
        print("There are", len(self.synapses), "layers of synapses")
        for n, layer in enumerate(self.layers):
            n += 1
            print("Layer", n, "-")
            print(layer)

    def add_layer(self, name, size):
        self.names.append(name)
        globals()[name] = size
        self.layer_count += 1
        
    def nonlin(self, x, deriv=False):
        #print("Derive:", deriv)
        if (deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def initialize(self, X, y, verbose = False):
        self.X = X
        self.y = y
        
        np.random.seed(seed = 68)

        self.synapses.append(2*np.random.random((self.inp,globals()[self.names[0]])) - 1)

        if self.layer_count == 1:
            for i in range(len(self.names)):
                self.synapses.append(2*np.random.random((globals()[self.names[i]], globals()[self.names[i]])) - 1)
        elif self.layer_count > 1:
            for i in range(len(self.names)-1):
                self.synapses.append(2*np.random.random((globals()[self.names[i]], globals()[self.names[i+1]])) - 1)                

        self.synapses.append(2*np.random.random((globals()[self.names[-1]], 1)) - 1)

        if verbose == True:
            print("SYNAPSES:")
            for syn in self.synapses:
                print(syn)
        
        for _ in range(self.layer_count+1):
            self.layers.append([])
            self.layersΔ.append([])
    
    def feedfoward(self, verbose = False):
        self.layers[0] = self.nonlin(np.dot(self.X, self.synapses[0]))
        for n, syn in enumerate(self.synapses):
            if(n > 0):
                self.layers[n] = self.nonlin(np.dot(self.layers[n-1], syn))
        
        '''
        if verbose == True:
            print("\nForeward propegation:")
            for n,layer in enumerate(self.layers):
                n += 1
                #print("LAYER",n,"\n", layer)
                print("LAYER",n,"\n", layer.shape)
        '''
        
        return self.layers[-1]
    
    def backprop(self, verbose = False):
        error = self.y - self.layers[-1]
        self.layersΔ[-1] = error * self.nonlin( self.layers[-1], True)

        #self.layersΔ[-2] = self.layersΔ[-1].dot(self.synapses[2].T) * self.nonlin( self.layers[-2], True)
        #self.layersΔ[-3] = self.layersΔ[-2].dot(self.synapses[1].T) * self.nonlin( self.layers[-3], True)

        for n in range(len(self.layers)-1):
            i = -1 * (n + 1)
            n = n+1
            n = len(self.layers) - n
            #print(self.layersΔ[i].shape, self.synapses[n].shape)
            temp = self.layersΔ[i].dot(self.synapses[n].T)
            self.layersΔ[i-1] = temp * self.nonlin( self.layers[i-1], True)            

        '''
        for n in range(len(self.layers)-1):
            i = -1 * (n + 1)
            n = n+1
            print(len(self.layers)-n, i, i-1)
        '''
        
        if verbose == True:
            print("\nBackward propegation:")
            for n,layer in enumerate(self.layersΔ):
                n += 1
                #print("LAYER",n,"\n", layer)
                print("LAYER",n,"\n", layer.shape)

    def update(self, verbose = True):
        
        #self.synapses[2] += self.layers[1].T.dot(self.layersΔ[2])
        #self.synapses[1] += self.layers[0].T.dot(self.layersΔ[1])
        
        for n in range(len(self.layers)-1):
            n = n+1
            n = len(self.layers) - n
            self.synapses[n] += self.lr * self.layers[n-1].T.dot(self.layersΔ[n])
        
        self.synapses[0] += self.lr * self.X.T.dot(self.layersΔ[0])
        
        '''
        if verbose == True:
            print("Synapses:")
            for n,layer in enumerate(self.synapses):
                print("LAYER",n,"\n", layer)
        '''        
        #syn2 += l2.T.dot(l3_delta)
        #syn1 += l1.T.dot(l2_delta)
        #syn0 += X.T.dot(l1_delta)
            
    def epoch(self):
        self.feedfoward()
        self.backprop()
        self.update()
                
    def predict(self, X, mode = 'standard'):
        
        layers = []
        for _ in range(self.layer_count+1):
            layers.append([])
            
        layers[0] = self.nonlin(np.dot(X, self.synapses[0]))
        for n, syn in enumerate(self.synapses):
            if(n > 0):
                layers[n] = self.nonlin(np.dot(layers[n-1], syn))
        
        '''
        if verbose == True:
            print("\nForeward propegation:")
            for n,layer in enumerate(self.layers):
                n += 1
                #print("LAYER",n,"\n", layer)
                print("LAYER",n,"\n", layer.shape)
        '''
        
        if mode == 'percent':
            preds = layers[-1]
        else:
            preds = np.round(layers[-1])
        
        return preds