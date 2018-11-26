import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

# parameter setting
num_input_frames = 10
num_train_samples = 1000 # best 1000
num_test_samples = 168
batch_size = 80 # best 80
epoch = 300 # best 300
# DATA_PATH = '/Users/lianjian/Desktop/project/cnn_attention/sample'
DATA_PATH = '../sample'


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(num_input_frames, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 700)
        self.fc2 = nn.Linear(700, 30 * 30)

        self.weight1 = nn.Linear(num_input_frames, 1)
        self.weight2 = nn.Linear(2, 1)

    def forward(self, x):

        # print("x size is", x.size())
        # give weights to each frame (get attention)
        y = self.get_attention(x)

        # convolution on x
        # print("x size is ", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # print(x.size())

        # linear combination of x and y

        x = torch.reshape(x, (-1, 900, 1))
        y = torch.reshape(y, (-1, 900, 1))

        z = torch.cat((x, y), 2)

        z = self.weight2(z).squeeze(-1)

        # print(z.size())

        return z



    def get_attention(self, x):
        y = x.permute(0, 2, 3, 1)
        y = self.weight1(y)
        y = y.permute(0, 3, 1, 2)

        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train():

    # get data and label

    training_features, training_labels = trainloader(n_samples=num_train_samples,
                                               n_input_frames=num_input_frames,
                                               data_path=DATA_PATH)

    # print(training_feature.shape)

    # get net instance
    net = Net()

    # optimization
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=1e-5,
                          momentum=0.9,
                          weight_decay=1e-5)

    # train the network
    # for every epoch
    for i in range(epoch):

        running_loss = 0.0

        # for every batch
        for j in range(int(num_train_samples / batch_size)):

            # extract training_feature and convert to tensors
            inputs = torch.from_numpy(training_features[batch_size * j:batch_size * (j + 1),::]).float()

            # extract training_labels and convert to tensors
            labels_temp = training_labels[batch_size * j:batch_size * (j + 1),::]
            labels_temp = np.reshape(labels_temp, (batch_size, 30 * 30))
            labels = torch.from_numpy(labels_temp).float()
            # labels_temp = np.reshape(labels_temp, (30 * 30, 1))
            # labels = torch.from_numpy(labels_temp).double()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #print running_loss
            running_loss += loss.item()
            if j % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (i + 1, j + 1, running_loss / 10))
                running_loss = 0.0

    torch.save(net.state_dict(), "net_para_1.pt")
    print("finish training")


    # # training performance
    # # print the result
    # inputs_test = torch.from_numpy(training_features[3:4,::]).float()
    # outputs_test = net(inputs_test)
    # # print(outputs_test.size())
    # outputs_test_mat = outputs_test.detach().numpy()
    # # print(outputs_test_mat.shape)
    #
    # outputs_test_mat = np.reshape(outputs_test_mat, (30, 30))
    #
    # # print(outputs_test_mat.shape)
    # outputs_test_mat[outputs_test_mat < 90] = 0
    #
    # plt.matshow(outputs_test_mat)
    # plt.show()
    #
    # # print the groundtruth
    # labels_test = torch.from_numpy(training_labels[3:4,::]).float()
    # labels_test_mat = labels_test.detach().numpy()
    # labels_test_mat = np.reshape(labels_test_mat, (30, 30))
    #
    # plt.matshow(labels_test_mat)
    # plt.show()


def test():
    # test
    test_features, test_labels = testloader(n_samples=num_test_samples,
                                               n_input_frames=num_input_frames,
                                               data_path=DATA_PATH)
    # print(test_features.shape)
    # print(test_labels.shape)

    net = Net()
    net.load_state_dict(torch.load("net_para_1.pt"))
    criterion = nn.MSELoss()

    # compute test_loss
    # print(outputs.size())
    # print(test_labels_tensor.size())

    running_test_loss = 0

    for i in range(num_test_samples):

        # convert test_features to tensors
        test_inputs = np.reshape(test_features[i,::], (1, num_input_frames, 30, 30))
        test_inputs_tensor = torch.from_numpy(test_inputs).float()

        # print(test_inputs_tensor.size())

        # transform labels
        test_labels_temp = np.reshape(test_labels[i,::], (1, 30 * 30))
        test_labels_tensor = torch.from_numpy(test_labels_temp).float()

        # print(test_labels_tensor.size())

        # feed the model with test_features
        # print(test_inputs_tensor.size())
        test_outputs = net(test_inputs_tensor)

        # compute test_loss_temp
        loss_temp = criterion(test_outputs, test_labels_tensor)

        running_test_loss += loss_temp

        print("sample", i + 1, "has loss", loss_temp.item())

    test_loss = running_test_loss / num_test_samples

    print("average test loss is: ", test_loss.item())

    # # draw the result
    # test_labels_mat = np.reshape(test_labels, (30, 30))
    # test_outputs_mat = np.reshape(test_outputs.detach().numpy(), (30, 30))
    # #
    # # print(test_outputs_mat)
    # # print(test_labels_mat)
    #
    # test_outputs_mat[test_outputs_mat < 90] = 0
    # test_outputs_mat[test_outputs_mat >= 90] = 255
    #
    # # plot the first image
    # plt.matshow(test_labels_mat)
    # plt.matshow(test_outputs_mat)
    # plt.show()




def trainloader(n_samples, n_input_frames, data_path):


    training_feature = np.zeros((n_samples, n_input_frames, 30, 30))
    training_label = np.zeros((n_samples, 1, 30, 30))

    for i in range(n_samples):

        for j in range(n_input_frames):

            # training feature, every 10 frames
            pth_feature = os.path.join(data_path,  str(i * n_input_frames + j) + '.png')
            img_feature = PIL.Image.open(pth_feature)
            img_feature_mat = np.asmatrix(img_feature)
            img_feature_mat = np.reshape(img_feature_mat, (1, 30, 30)) / 255

            training_feature[i,j,::] = img_feature_mat

            # print(img_feature_mat.shape)

        # training labels, the 11th frame
        pth_label = os.path.join(data_path, str(i * n_input_frames + n_input_frames) + '.png')
        img_label = PIL.Image.open(pth_label)
        img_label_mat = np.asmatrix(img_label)
        img_label_mat = np.reshape(img_label_mat, (1, 30, 30)) / 255

        training_label[i, 0, ::] = img_label_mat

        # print(img_label_mat.shape)




    # x = torch.from_numpy(training_feature)
    # y = torch.from_numpy(training_label)
    # a = torch.from_numpy(feat_array)

    return training_feature, training_label
    # return x, y


def testloader(n_samples, n_input_frames, data_path):


    test_feature = np.zeros((n_samples, n_input_frames, 30, 30))
    test_label = np.zeros((n_samples, 1, 30, 30))

    for i in range(n_samples):

        for j in range(n_input_frames):

            # testing features, extract after the training features which is num_input_frames * num_train_samples + 1
            pth_feature = os.path.join(data_path,  str(i * n_input_frames + j + n_input_frames * num_train_samples + 1) + '.png')
            img_feature = PIL.Image.open(pth_feature)
            img_feature_mat = np.asmatrix(img_feature)
            img_feature_mat = np.reshape(img_feature_mat, (1, 30, 30))

            test_feature[i,j,::] = img_feature_mat / 255

            # print(img_feature_mat.shape)

        # testing labels, the 11th frame, extract after the training labels which is num_input_frames * num_train_samples + 1
        pth_label = os.path.join(data_path, str(i * n_input_frames + n_input_frames + n_input_frames * num_train_samples + 1) + '.png')
        img_label = PIL.Image.open(pth_label)
        img_label_mat = np.asmatrix(img_label)
        img_label_mat = np.reshape(img_label_mat, (1, 30, 30)) / 255

        test_label[i, 0, ::] = img_label_mat

        # print(img_label_mat.shape)




    # x = torch.from_numpy(training_feature)
    # y = torch.from_numpy(training_label)
    # a = torch.from_numpy(feat_array)

    return test_feature, test_label
    # return x, y



if __name__ == '__main__':
    train()
    test()