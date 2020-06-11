import os
import os.path
import torchvision
from torchvision import transforms

import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch import nn, optim
import torch.utils.data
from gcommand_loader import GCommandLoader

# for google colab
device = torch.device("cuda" if torch.cuda else "cpu")

# defines
TRAIN_PATH = "./data/train"
VALID_PATH = "./data/valid"
TEST_PATH = "./data/test"
TEST_Y_FILE = "test_y"
NORM1 = 0.1307
NORM2 = 0.3081

# hyper parameters and parameters
epochs = 7
eta = 0.001
batch_size = 100
tag_amount = 30

# transforms on the data (compose, normalize etc)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((NORM1,), (NORM2,))])


# this class creates the convolution network
class ConvolutionModel(nn.Module):
    # constructor
    def __init__(self):
        super(ConvolutionModel, self).__init__()
        # build layers
        self.firstLay = self.build_one_layer(1, 16)
        self.secondLay = self.build_one_layer(16, 64)
        self.thirdLay = self.build_one_layer(64, 15)
        # build drop out
        self.drop_out = nn.Dropout()
        # build linear transformation to change the size of the data
        self.lt1 = nn.Linear(3600, 150)
        self.lt2 = nn.Linear(150, 50)
        self.lt3 = nn.Linear(50, 30)

    # building one layer of the convolution network
    def build_one_layer(self, num1, num2):
        layer = nn.Sequential(
            nn.Conv2d(num1, num2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        return layer

    # forward propagation
    def forward(self, x):
        fwd = self.firstLay(x)
        fwd = self.secondLay(fwd)
        fwd = self.thirdLay(fwd)
        fwd = fwd.reshape(fwd.size(0), -1)
        # apply drop out
        fwd = self.drop_out(fwd)
        # apply linear transformations
        fwd = self.lt1(fwd)
        fwd = self.lt2(fwd)
        fwd = self.lt3(fwd)
        return fwd


# training the model
def train(model, train_loader, loss_func, optimizer):
    loss_arr = []
    correct_avg_arr = []
    train_length = len(train_loader)

    for epoch in range(epochs):
        for i, (examples, tags) in enumerate(train_loader):
            examples, tags = examples.to(device), tags.to(device)
            # predictions- run forward
            y_hats = model(examples)

            # loss calculation
            loss = loss_func(y_hats, tags)
            loss_arr.append(loss.item())

            # run back propagation using optimizer (which is set to Adam in main())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the accuracy of the model- how many predictions are correct out of all predictions
            all_predicts = tags.size(0)
            _, predicted = torch.max(y_hats.data, 1)
            correct_predicts = (predicted == tags).sum().item()
            correct_avg_arr.append(correct_predicts / all_predicts)



# cross validation process on our model
def validate(model, valid_loader):
    model.eval()
    with torch.no_grad():
        correct_predicts = 0
        all_predicts = 0
        for examples, tags in valid_loader:
            examples, tags = examples.to(device), tags.to(device)
            y_hats = model(examples)
            _, predicted = torch.max(y_hats.data, 1)
            all_predicts += tags.size(0)
            correct_predicts += (predicted == tags).sum().item()

# test the model
def test(model, test_loader, test_set):
    model.eval()
    predicts_arr = []
    with torch.no_grad():
        for tags, _ in test_loader:
            tags = tags.to(device)
            _ = _.to(device)
            # predict y_hat and save it in an array
            y_hats = model(tags)
            _, predict = torch.max(y_hats.data, 1)
            predicts_arr.extend(predict)
        # save results (y_hats) in a file with the format required
        with open(TEST_Y_FILE, "w") as file:
            for spect, prediction in zip(test_set.spects, predicts_arr):
                file.write("{}, {}".format(os.path.basename(spect[0]), str(prediction.item()) + '\n'))


# loading data
def data_loading(address, is_shuffle):
    cmd_load = GCommandLoader(address)
    data_loader = torch.utils.data.DataLoader(cmd_load, batch_size=100, shuffle=is_shuffle,
                                              num_workers=20, pin_memory=True, sampler=None)
    return data_loader


# main function
def main():
    # loading data (train, valid and test)
    train_loader = data_loading(TRAIN_PATH, True)
    valid_loader = data_loading(VALID_PATH, True)
    test_loader = data_loading(TEST_PATH, False)

    # create our model - the convolution network
    model = ConvolutionModel().to(device)

    # create loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    # train, validate and test
    train(model, train_loader, loss_func, optimizer)
    validate(model, valid_loader)
    #test(model, test_loader, test_set)
    test(model, test_loader, GCommandLoader(TEST_PATH))


if __name__ == '__main__':
    main()
