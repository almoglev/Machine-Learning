import sys
import numpy as np
from scipy.stats import mstats

"""
normal the arguments in the matrix 
"""
"""
def normal_x_train(train_x, test_x):
    #transpose the matrix to work on the col
    to_col = np.transpose(train_x)
    to_col_test = np.transpose(test_x)
    for i in range(len(to_col)):
        row = to_col[i]
        if i<len(to_col_test):
            row_test = to_col_test[i]
        min = row.min()
        max = row.max()
        to_div = max - min
        for j in range(len(row)):
            if to_div != 0:
                to_col[i][j] = (row[j] - min) / to_div
                if(j<len(to_col_test)):
                    to_col_test[i][j] = (row_test[j] - min) / to_div
            else:
                to_col[i][j] = 1 / len(train_x)
                if (j < len(to_col_test)):
                    to_col_test[i][j] = 1/len(test_x)
    #transpose back
    train_x = np.transpose(to_col)
    return train_x
"""
"""
normal the arguments according to zscore function
"""
"""
def normal_zscore(train_x):
    to_col = np.transpose(train_x)
    for i in range(len(to_col)):
        to_col[i] = mstats.zscore(to_col[i])
    train_x = np.transpose(to_col)
    return train_x
"""

"""
insert the data into matrix
"""
def to_matrix(file):
    #read the data from the file
    get_file_value = open(file, 'r')
    x_train = np.array([[]])
    flag_first = 0
    line = (get_file_value.readline())
    #read line by line
    while line:
        i = 0
        m = line[i]
        num = ""
        temp_x_train = np.array([])
        while ((m != '\n') & (i < len(line) - 1)):
            #get the sex
            if (m == 'M'):
                temp_x_train = np.append(temp_x_train, 0.25)
            elif (m == 'F'):
                temp_x_train = np.append(temp_x_train, 0.5)
            elif (m == 'I'):
                temp_x_train = np.append(temp_x_train, 0.75)
            else:
                while ((m != ",") & (i < len(line) - 1)):
                    num += m
                    i += 1
                    m = line[i]
                if (num != ""):
                    temp_x_train = np.append(temp_x_train, float(num))
            if (i + 1 == len(line)):
                break
            i += 1
            m = line[i]
            num = ""
        temp1 = np.array([temp_x_train])
        if (flag_first != 0):
            x_train = np.concatenate((x_train, temp1))
        else:
            x_train = np.copy(temp1)
            flag_first = 1
        line = (get_file_value.readline())
    return x_train

"""
get the data from the file with y values
"""
def read_y_train(file):
    #read the data
    get_file_value = open(file, 'r').readlines()
    arr = []
    for line in get_file_value:
        arr.append(int(float(line.replace("\n", ""))))
    return arr

"""
run the test file
"""
def predict(test_x, perceptron_w, svm_w, pa_w):
    # loop through the examples
    for x in test_x:
        # predictions for each algorithm
        perceptron_y_hat = np.argmax(np.dot(x, np.transpose(perceptron_w)))
        svm_y_hat =  np.argmax(np.dot(x, np.transpose(svm_w)))
        pa_y_hat =  np.argmax(np.dot(x, np.transpose(pa_w)))
        print("perceptron: " + str(perceptron_y_hat) +", svm: "+str(svm_y_hat)+", pa: " + str(pa_y_hat))

"""
perceptron algorithem
"""
def perceptron(x_info, y_info):
    #x_info = normal_x_train(x_info)
    # preparation
    m = len(x_info)
    # initialise eta and weight vector
    eta = 0.1
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0

    epochs = 15
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))
            # update w in case our predication is wrong
            if y_hat != y:
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
                bad_y_hat = bad_y_hat + 1
        #print("preceptron err = " + str(float(bad_y_hat) / m))
        err_avg = float((bad_y_hat) / m)
        bad_y_hat = 0
        eta = eta / (e + 100)
    return w


"""
passive agressive algorithem
"""
def pa(x_info, y_info):
    #x_info = normal_zscore(x_info)
    #x_info = normal_x_train(x_info)
    # preparation
    m = len(x_info)

    # initialise tau and weight vector
    tau = 0
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0

    epochs = 30
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))

            if y_hat != y:
                bad_y_hat = bad_y_hat + 1
                y_hat = int(y_hat)
                y = int(y)

                # calculate tau (by calculating hinge loss function and the norm of x powered by 2)
                hinge_loss = max(0.0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))
                x_norm = (2*(np.power(np.linalg.norm(x, ord=2), 2)))
                # update w in case our predication is wrong
                if (x_norm != 0):
                    tau = (hinge_loss / x_norm)
                    w[y, :] = w[y, :] + tau * x
                    w[y_hat, :] = w[y_hat, :] - tau * x

        err_avg = float((bad_y_hat) / m)
        #print("pa err = " + str(float(bad_y_hat) / m))
        bad_y_hat = 0
    return w

"""
SVM algorithem
"""
def svm(x_info, y_info):
    #x_info = normal_x_train(x_info)
    # preparation
    m = len(x_info)
    # initialise eta, lamda and weight vector
    eta = 0.1
    lamda= 0.0001
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0

    epochs = 10
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))
            # update w in case our predication is wrong
            if y_hat != y:
                w[y, :] = ((1-(eta*lamda)) * w[y, :]) + eta * x
                w[y_hat, :] = (1-eta*lamda) * w[y_hat, :] - eta * x
                w[(3 - (y+y_hat)), :] = (1-eta*lamda)*w[(3 - (y+y_hat)), :]
                bad_y_hat = bad_y_hat + 1
            else:
                # in svm we update w also when the prediction is correct
                for i in range(0,len(w)):
                    if (i != y):
                        w[i, :] = (1 - (eta * lamda)) * w[i, :]

        err_avg = float((bad_y_hat) / m)
        bad_y_hat = 0
        eta = eta / (e + 100)
    return w

"""
main function
"""
def main():
    if len(sys.argv) != 4:
        print("not enough arguments\n")
        sys.exit(-1)
    else:
        arg1 = sys.argv[1]
        x_train = to_matrix(arg1)
        arg3 = sys.argv[3]
        test_x = to_matrix(arg3)
        #x_train = normal_x_train(x_train, test_x)

        arg2 = sys.argv[2]
        y_train = read_y_train(arg2)

        perceptron_w = perceptron(x_train, y_train)
        svm_w = svm(x_train, y_train)
        pa_w = pa(x_train, y_train)

        predict(test_x, perceptron_w, svm_w, pa_w)

main()
