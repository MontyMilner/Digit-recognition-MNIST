import numpy as np
from mnist import MNIST
from matplotlib import pyplot as plt

def init_params():
    image_length=784
    W1=np.random.rand(10,image_length)-0.5
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1=W1.dot(X) + b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1) + b2
    A2=softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    OHY=one_hot(Y)
    dZ2=2*(A2-OHY)
    dW2=1/samples*dZ2.dot(A1.T)
    db2=1/samples*np.sum(dZ2,1)
    dZ1=W2.T.dot(dZ2)*deriv_ReLU(Z1)
    dW1=1/samples*dZ1.dot(X.T)
    db1=1/samples*np.sum(dZ1,1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size
    
def ReLU(array):
    return np.maximum(array,0)

def deriv_ReLU(array):
    return array > 0

def softmax(array):
    exp = np.exp(array - np.max(array)) 
    return exp / exp.sum(axis=0)

def one_hot(array):
    Y=np.zeros((array.size,array.max()+1))
    Y[np.arange(array.size),array]=1
    return Y.T


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i%10 == 0:
            print("Iteration: ",i)
            predictions=get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions,Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



###______MAIN_____###

mndata = MNIST('samples')
images, labels = mndata.load_training()
samples=10005
scale_factor=255



X_train=(np.array(images[:samples]).T)/scale_factor
Y_train=np.array(labels[:samples])
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)
test_prediction(10000, W1, b1, W2, b2)
test_prediction(7001, W1, b1, W2, b2)
test_prediction(1002, W1, b1, W2, b2)
test_prediction(801, W1, b1, W2, b2)
test_prediction(54, W1, b1, W2, b2)


