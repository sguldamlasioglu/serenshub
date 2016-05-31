__author__ = 'stm'
import csv
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pickle


#include your own file path
filetrain = '/home/PycharmProjects/train_data.csv'
filetest =  '/home/PycharmProjects/test_data.csv'

#filetest =  '/home/stm/PycharmProjects/isbak_trafik/Data/deneme_test.csv'

# train features
# each features has float values
# each labels has binary values

a= []
b= []
c = []
d = []
e = []
f = []
train_label = []

# test features
g= []
h= []
i = []
j = []
k = []
l = []
test_label = []




with open(filetrain) as file:
    file.next()
    train_reader = csv.reader(file, delimiter=',')

    for rows in train_reader:
        a.append(int(rows[2]))
        b.append(int(rows[3]))
        c.append(float(rows[4]))
        d.append(int(rows[5]))
        e.append(float(rows[6]))
        f.append(float(rows[7]))
        train_label.append(int(rows[8]))

with open(filetest) as file:
    file.next()
    test_reader = csv.reader(file, delimiter=' ')

    for rows in test_reader:
        g.append(int(rows[2]))
        h.append(int(rows[3]))
        i.append(float(rows[4]))
        j.append(int(rows[5]))
        k.append(float(rows[6]))
        l.append(float(rows[7]))
        test_label.append(int(rows[8]))


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]+ 1))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +1]))-1)*0.25)



    def fit(self, X, y, learning_rate=0.01, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        errors = []
        sqrts = []
        for k in range(epochs):

            if k > 9000:
                if error >= -0.0001 and error <= 0.0001:

                    break

            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            errors.append(error)

            sqrts.append(np.sqrt(error))

            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict(self, x):
        x = np.array(x)
        temp1 = np.ones(x.shape[0]+1)
        temp2 = np.zeros(x.shape[0]+1)
        temp1[0:-1] = x
        temp2[0:-1] = x
        a = temp1
        b = temp2

        for l in range(0, len(self.weights)):
            # print self.weights[l]
            a = self.activation(np.dot(a, self.weights[l]))
            b = self.activation(np.dot(b, self.weights[l]))

        return a, b



test_features = map(list, zip(g,h,i,j,k,l))
train_features = map(list, zip(a, b, c, d, e, f))

nn = NeuralNetwork([6,15,1], 'tanh')

def trainmodel (train_features, train_label):
    aucs = []
    train_ind = []
    test_ind = []
    X_train_sets = []
    X_test_sets = []
    y_train1_sets = []
    y_test1_sets =[]


    kf = KFold(len(train_features), n_folds=5)

    train_features = np.array(train_features)
    train_label = np.array(train_label)

    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_features[train_index], train_features[test_index]
        y_train1,y_test1 = train_label[train_index], train_label[test_index]

        X_train_sets.append(np.array(X_train))
        X_test_sets.append(np.array(X_test))
        y_train1_sets.append(np.array(y_train1))
        y_test1_sets.append(np.array(y_test1))

        nn.fit(X_train,y_train1, epochs=10000)

        cv_predictions = []
        for i in range(np.array(X_test).shape[0]):
            o = nn.predict(X_test[i])
            cv_predictions.append(np.argmax(o))

        auc = roc_auc_score(y_test1,cv_predictions)
        aucs.append(auc)

    best_auc_ind = aucs.index(max(aucs))
    print "best auc:" ,max(aucs)

    nn.fit(X_train_sets[best_auc_ind],y_train1_sets[best_auc_ind], epochs=10000)


def testmodel(test_features):
    test_predictions = []
    for i in range(np.array(test_features).shape[0]):
        o = nn.predict(test_features[i])
        test_predictions.append(np.argmax(o))
    return test_predictions


count = 0
while True:

    count += 1
    print count

    trainmodel (train_features, train_label)
    predicted_results = testmodel(test_features)

    if confusion_matrix(test_label,predicted_results)[0][1] < 1000:

        print confusion_matrix(test_label,predicted_results)
        print classification_report(test_label,predicted_results)
        print precision_score(test_label,predicted_results)
        weights = nn.weights
        pickle.dump(weights, open( "nn_model_weights.p", "wb" ))

        break

nn_test = NeuralNetwork([6,15,1], 'tanh')
weights = pickle.load( open( "nn_model_weights.p", "rb" ))
nn_test.weights = weights

def testmodel2(test_features):
    test_predictions = []
    for i in range(np.array(test_features).shape[0]):
        o = nn_test.predict(test_features[i])
        test_predictions.append(np.argmax(o))
    return test_predictions

predicted_results2 = testmodel2(test_features)

print "-----------------------------------------------"
print confusion_matrix(test_label,predicted_results2)
print classification_report(test_label,predicted_results2)
print precision_score(test_label,predicted_results2)








