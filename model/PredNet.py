import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from model.DiscriminativeCell import DiscriminativeCell
from model.GenerativeCell import GenerativeCell


# Define some constants
OUT_LAYER_SIZE = (1,) + tuple(2 ** p for p in range(4, 10))
ERR_LAYER_SIZE = tuple(size * 2 for size in OUT_LAYER_SIZE)
IN_LAYER_SIZE = (1,) + ERR_LAYER_SIZE

print (OUT_LAYER_SIZE, ERR_LAYER_SIZE, IN_LAYER_SIZE)

def data_loader(data_path, n_samples, n_frames):
    feature_data = np.zeros((n_samples, n_frames, 1, 32, 32))
    label_data   = np.zeros((n_samples, n_frames, 1, 32, 32))


    for i in range(n_samples):
        rd_start_frame = np.random.randint(0, n_samples-n_frames)
        for j in range(n_frames):
            pth = os.path.join(data_path, str(rd_start_frame + j) + '.png')
            feat_im = Image.open(pth)
            feat_array = np.asarray(feat_im)
            feat_array = np.reshape(feat_array, (1, 32, 32))
            feat_array.setflags(write=1)
            feat_array[feat_array >= 1] = 1
            feature_data[i, j, ::] = feat_array

            label_im = Image.open(os.path.join(data_path, str(rd_start_frame + n_frames + 1) + '.png'))
            label_array = np.asarray(label_im)
            label_array = np.reshape(label_array, (1, 32, 32))
            label_array.setflags(write=1)
            label_array[label_array >= 1] = 1
            label_data[i, j, ::] = label_array
        print('Proccessing %d sample' % i)
    return feature_data, label_data

class PrednetModel(nn.Module):
    """
    Build the Prednet model
    """

    def __init__(self, error_size_list):
        super().__init__()
        self.number_of_layers = len(error_size_list)
        for layer in range(0, self.number_of_layers):
            setattr(self, 'discriminator_' + str(layer + 1), DiscriminativeCell(
                input_size={'input': IN_LAYER_SIZE[layer], 'state': OUT_LAYER_SIZE[layer]},
                hidden_size=OUT_LAYER_SIZE[layer],
                first=(not layer)
            ))
            setattr(self, 'generator_' + str(layer + 1), GenerativeCell(
                input_size={'error': ERR_LAYER_SIZE[layer], 'up_state':
                    OUT_LAYER_SIZE[layer + 1] if layer != self.number_of_layers - 1 else 0},
                hidden_size=OUT_LAYER_SIZE[layer],
                error_init_size=error_size_list[layer]
            ))

    def forward(self, bottom_up_input, error, state):

        # generative branch
        up_state = None
        for layer in reversed(range(0, self.number_of_layers)):
            state[layer] = getattr(self, 'generator_' + str(layer + 1))(
                error[layer], up_state, state[layer]
            )
            up_state = state[layer][0]

        # discriminative branch
        for layer in range(0, self.number_of_layers):
            if layer:
                error[layer], _ = getattr(self, 'discriminator_' + str(layer + 1))(
                    error[layer - 1],
                    state[layer][0]
                )
            else:
                error[layer], prediction = getattr(self, 'discriminator_' + str(layer + 1))(
                    bottom_up_input,
                    state[layer][0]
                )

        return error, state, prediction

def _test_training():
    number_of_layers = 3
    T = 1000  # sequence length
    max_epoch = 1000  # number of epochs
    lr = 1e-4      # learning rate
    Batch_Size = 10

    DATA_PATH = '../data/'
    data, label = data_loader(DATA_PATH, T, 1)
    # set manual seed
    torch.manual_seed(0)

    L = number_of_layers - 1
    print('\n---------- Train a', str(L + 1), 'layer network ----------')
    print('Create the input image and target sequences')
    # input_sequence = Variable(torch.rand(T, 1, 3, 15 * 2 ** L, 15 * 2 ** L))
    input_sequence = Variable(torch.tensor(data, dtype=torch.float))
    print('Input has size', list(input_sequence.data.size()))

    error_init_size_list = tuple(
        (1, ERR_LAYER_SIZE[l], 8 * 2 ** (L - l), 8 * 2 ** (L - l)) for l in range(0, L + 1)
    )
    print('The error initialisation sizes are', error_init_size_list)
    target_sequence = Variable(torch.zeros(T, *error_init_size_list[0]))

    print('Define a', str(L + 1), 'layer Prednet')
    model = PrednetModel(error_init_size_list)

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss(reduction='elementwise_mean')

    print('Run for', max_epoch, 'iterations')
    last_loss = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    for epoch in range(0, max_epoch):
        state = [None] * (L + 1)
        error = [None] * (L + 1)
        loss = 0
        time_weight = 0
        for t in range(0, T):
            error, state, prediction = model(input_sequence[t], error, state)
            loss += time_weight * loss_fn(error[0], target_sequence[t])
            time_weight = 1/(Batch_Size - 1)
            if (t + 1) % Batch_Size == 0:
                state = [None] * (L + 1)
                error = [None] * (L + 1)
                time_weight = 0

        loss = loss*Batch_Size/T
        if epoch == max_epoch - 1:
            print ('original image is:', plt.imshow(np.reshape(input_sequence[t].numpy(), (32, 32))))
            plt.show()
            print ('predicted image is:', plt.imshow(np.reshape(prediction.detach().numpy(), (32, 32))))
            plt.show()
        print(' > Epoch {:2d} loss: {:.5f} delta loss: {:.6f} learn rate : {:.6f}'.format((epoch + 1), loss.data, abs(last_loss - loss), lr))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        optimizer.step()

        # learning_rate step against the gradient
        # for p in model.parameters():
        #    p.data.sub_(p.grad.data * lr)
        if (epoch + 1) % 100 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        last_loss = loss


def _main():
    #_test_one_layer_model()
    # _test_two_layer_model()
    # _test_L_layer_model()
    _test_training()


if __name__ == '__main__':
    _main()


__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Prototype"  # "Prototype", "Development", or "Production"
__date__ = "Feb 17"