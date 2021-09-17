from anyLayerNN2 import *
from MNIST01 import *
from sklearn.metrics import accuracy_score


X_train, y_train, X_test, y_test = get_MNIST()
        
nn = NN(784)
nn.add_layer("one", 1)
nn.add_layer("two", 2)
#nn.add_layer("three", 28)
#nn.add_layer("four", 28)
#nn.add_layer("five", 28)
#nn.add_layer("six", 28)

nn.initialize(X_train, y_train)

epochs = 100

for _ in range(epochs):
    nn.epoch()  
    
preds = nn.predict(X_test)

print(accuracy_score(preds, y_test))

preds = nn.predict(X_test, mode = 'percent')
print("\nThe prediction is:")
print(preds)

