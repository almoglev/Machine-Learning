import numpy as np
import scipy

"""
defines
"""
IMAGE_SIZE = 784
FIRST_LAYER_SIZE = 150
LAST_LAYER_SIZE = 10
ETA = 0.005
EPOCH = 50
MAX_COLOR = 255
TRAIN_PERCENT = 0.8


"""
activation function- relu (if the value<=0 returns 0, otherwise returns the value)
"""
def relu(x):
    return np.maximum(np.zeros(np.shape(x)), x)


"""
derivative of relu (for back propagation)- if value<= returns 0, otherwise returns 1)
"""
def relu_derivative(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


"""
softmax function- to normalize the last layer
"""
def softmax(x):
    exps = np.exp(x)
    sum_exps = np.sum(exps)
    # to make sure we don't divide by 0
    if sum_exps == 0:
        sum_exps = 0.001
    return exps / sum_exps


"""
initializing parameters (weights matrix and bias vectors) with random values
"""
def init_params():
    w1 = (0.2 * np.random.rand(FIRST_LAYER_SIZE, IMAGE_SIZE)) - 0.1
    b1 = (0.2 * np.random.rand(FIRST_LAYER_SIZE, 1)) - 0.1
    w2 = (0.2 * np.random.rand(LAST_LAYER_SIZE, FIRST_LAYER_SIZE)) - 0.1
    b2 = (0.2 * np.random.rand(LAST_LAYER_SIZE, 1)) - 0.1
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return params


"""
the function is doing forward propagation on one image and returns
the updated weights and biases, the loss and the z's and h's.
"""
def forward(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x = np.array(x)
    # flattening the image from matrix to a vector
    x.shape = (IMAGE_SIZE, 1)
    # image is going through the layers
    z1 = np.dot(w1, x) + b1
    h1 = relu(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    # creating a vector of the real label- 1 in the index of real classification and 0's in the rest
    y_vector = np.zeros(LAST_LAYER_SIZE)
    y_vector[np.int(y)] = 1
    # calculating the loss
    loss = np.negative((np.dot(y_vector, np.log(h2))))
    updated_params = {'x': x, 'y_vector': y_vector, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        updated_params[key] = params[key]
    return updated_params

"""
the function is doing backward propagation on one image and returns
the derivatives of the the params
"""
def backward(forward_p):
    # the information recieved after forward propagation
    x, y_vector, z1, h1, z2, h2, w1, b1, w2, b2 = [forward_p[key] for key in (
        'x', 'y_vector', 'z1', 'h1', 'z2', 'h2', 'w1', 'b1', 'w2', 'b2')]
    # the real label vector (1 in real label index and 0'z in the rest)
    y_vector = np.asarray(y_vector)
    y_vector.shape = (LAST_LAYER_SIZE, 1)

    # derivatives
    dl_dz2 = np.subtract(h2, y_vector)
    dl_dw2 = np.dot(dl_dz2, np.transpose(h1))
    dl_db2 = dl_dz2
    dl_dh1 = np.dot(np.transpose(w2), dl_dz2)
    dl_dz1 = np.multiply(dl_dh1, relu_derivative(z1))
    dl_dw1 = np.dot(dl_dz1, np.transpose(x))
    dl_db1 = dl_dz1
    return {'w1': dl_dw1, 'b1': dl_db1, 'w2': dl_dw2, 'b2': dl_db2}

"""
shuffling the data - prevents overfitting
"""
def shuffle_data(data_x, data_y):
    zip_info = list(zip(data_x, data_y))
    np.random.shuffle(zip_info)
    zip_x, zip_y = zip(*zip_info)
    return [zip_x, zip_y]

"""
updating the parameters-
the function returns the updated parameters
"""
def update_params(params, back_p):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    dl_dw1, dl_db1, dl_dw2, dl_db2 = [back_p[key] for key in ('w1', 'b1', 'w2', 'b2')]
    w1 = np.subtract(w1, np.multiply(dl_dw1, ETA))
    b1 = np.subtract(b1, np.multiply(dl_db1, ETA))
    w2 = np.subtract(w2, np.multiply(dl_dw2, ETA))
    b2 = np.subtract(b2, np.multiply(dl_db2, ETA))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

"""
train process- training the model on the examples
the function returns the parameters with the loss of each example(image)
"""
def train(x, y, params):
    zip_x, zip_y = shuffle_data(x, y)
    loss_arr = []
    for image, label in zip(zip_x, zip_y):
        # normalizing image
        image = np.divide(image, MAX_COLOR)

        forward_p = forward(image, label, params)
        back_p = backward(forward_p)
        params = update_params(params, back_p)
        loss_arr.append(forward_p['loss'])
    return [params, np.mean(loss_arr)]

"""
the function receives a normalized image and params
and predicts y_hat for the image (classify the image)
the function returns the argmax of the prediction
"""
def predict_y_hat(image, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    # flattening the image to a vector
    image = np.array(image)
    image.shape = (IMAGE_SIZE, 1)
    # calculating and predicting y_hat
    z1 = np.dot(w1, image) + b1
    h1 = relu(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    return np.argmax(h2)

"""
cross validation process- the function receives images and labels (20% of the train)
and predicts y_hat.
the function returns the percent of correct predictions out of all predictions.
"""
def validation(x, y, params):
    good_y_hat = 0.0
    total_predict = 0.0
    for image, label in zip(x, y):
        # normalizing image
        image = np.divide(image, MAX_COLOR)
        # predict y_hat
        y_hat = predict_y_hat(image, params)

        total_predict += 1
        if label == y_hat:
            good_y_hat += 1
    return good_y_hat / total_predict

"""
the function receives test file and params and runs test procedure on the images.
the fucnction returns an array that contains the predictions (y_hats) of the images
"""
def test(test_x, params):
    # array that will store all predictions - this will be the output in the test_y.txt file
    predictions_arr = []
    for image in test_x:
        # normalizing the image
        image = np.divide(image, MAX_COLOR)
        predictions_arr.append(int(predict_y_hat(image, params)))
    return predictions_arr


"""
main function
"""
data_x = np.loadtxt("train_x")
data_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")

# shuffle images
zip_x, zip_y = shuffle_data(data_x, data_y)

# split train file to 80% training and 20% validation. examples are chosen rendomly
train_x_size = np.size(zip_y)
division = int(TRAIN_PERCENT * train_x_size)
# 80% to train
train_x = zip_x[:division]
train_y = zip_y[:division]
# 20% to cross validation
validation_x = zip_x[division:]
validation_y = zip_y[division:]

#initializing parameters
parameters = init_params()

for epoch in range(EPOCH):
    # train
    parameters, loss = train(train_x, train_y, parameters)
    # validation
    cross_valid = validation(validation_x, validation_y, parameters)
    """
    print("\n" + str(epoch))
    print("train loss: " + str(loss))
    print("validation correct: " + str(cross_valid))
    """


# test and save results to a file
test_labels = test(test_x, parameters)
np.savetxt("test_y", test_labels, fmt='%d', delimiter='\n')