import os

import dataset
import torch
import model
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoder


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train():
    learning_rate = 0.0001
    net = model.Net()
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    '''data'''
    datapath = ''
    predataset = dataset.preTrainDataset(datapath)
    dataloader = DataLoder(predataset,
                           batch_size=64,
                           num_workers=1,
                           shuffle=True,
                           drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = 0
    with tqdm.tqdm(total=100000, miniters=1, mininterval=0) as progress:
        while True:
            for inputs in dataloader:
                net.train()
                hr, lr = inputs[-1][0], inputs[-1][1]

                hr = hr.to(device)
                lr = lr.to(device)

                out = net(lr)
                loss = loss_fn(hr, out)

                progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                                         iter=step, loss=loss.item(), lr=learning_rate))

                progress.update()

                if step > 0 and step % 30000 == 0:
                    learning_rate = learning_rate / 10
                    adjust_learning_rate(optimizer, new_lr=learning_rate)
                    print("Learning rate reduced to {lr}".format(lr=learning_rate))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                optimizer.step()

                step += 1

            if step > 100000:
                print('Done training.')
                break

    save_path = os.path.join('./checkpoint','Pretrain.pth')
    torch.save(net.state_dict(), save_path)
    print("Model is saved !")

if __name__ == '__main__':
    train()




