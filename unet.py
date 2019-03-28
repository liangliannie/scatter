#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:17:05 2018

@author: bill
@updated: Liang
"""

import torch
from torch.autograd import Variable
import options_scatter as options
import os
from pytorch_msssim import MSSSIM
# # import sparseconvnet as scn
# import framelet
# import Nets
# import test2
import VGG

class UNet_down_block(torch.nn.Module):

    def __init__(self, input_channel, output_channel, down_sample):
        super(UNet_down_block, self).__init__()
        kernel_size = 3
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.down_sampling = torch.nn.Conv2d(input_channel, input_channel, kernel_size, stride=(2, 2), padding=(1, 1), bias=False)
        self.down_sample = down_sample


    def forward(self, x):
        if self.down_sample:
            x = self.down_sampling(x)
        x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        return x

class UNet_up_block(torch.nn.Module):

    def __init__(self, prev_channel, input_channel, output_channel, ID):
        super(UNet_up_block, self).__init__()
        kernel_size = 3
        self.ID = ID
        self.up_sampling = torch.nn.ConvTranspose2d(input_channel, input_channel, 4, stride=(2, 2), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)


    def forward(self, prev_feature_map, x):

        if self.ID == 1:
            x = self.up_sampling(x)
        elif self.ID == 2:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        elif self.ID == 3:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='area') #‘nearest’ | ‘linear’ | ‘bilinear’ | ‘trilinear’ | ‘area’
        x = torch.cat((x, prev_feature_map), dim=1)
        x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        return x


class UNet(torch.nn.Module):

    def __init__(self, opts):
        super(UNet, self).__init__()

        self.opts = opts
        input_channel_number = opts.input_channel_number
        output_channel_number = opts.output_channel_number
        # kernel_size = opts.kernel_size # we could change this later
        kernel_size = 3
        # Encoder network
        self.down_block1 = UNet_down_block(input_channel_number, 64, False) # 64*520
        self.down_block2 = UNet_down_block(64, 128, True) # 64*520
        self.down_block3 = UNet_down_block(128, 256, True) # 64*260


        # bottom convolution
        self.mid_conv1 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.mid_conv2 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.mid_conv3 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False) #, dilation=4 # 64*260
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.mid_conv4 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.mid_conv5 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn5 = torch.nn.BatchNorm2d(256)

        # Decoder network
        self.up_block2 = UNet_up_block(128, 256, 128, 1)# 64*520
        self.up_block3 = UNet_up_block(64, 128, 64, 1)# 64*520
        # Final output
        self.last_conv1 = torch.nn.Conv2d(64, 64, 3, padding=(1, 1), bias=False)# 64*520
        self.last_bn = torch.nn.BatchNorm2d(64) # 64*520
        self.last_conv2 = torch.nn.Conv2d(64, output_channel_number, 3, padding=(1, 1))# 64*520
        # self.linear1 = torch.nn.Sequential(*self.lin_tan_drop(64*520, 1024))
        # self.linear2 = torch.nn.Sequential(*self.lin_tan_drop(1024, 64*520))
        self.softplus = torch.nn.Softplus(beta=5, threshold=100)
        # self.softplus = torch.nn.ReLU()

    def lin_tan_drop(self, num_features_in, num_features_out, dropout=0.5):
        layers = []
        layers.append(torch.nn.Linear(num_features_in, num_features_out, bias=True))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Dropout(p=dropout))
        return layers

    def forward(self, x, test=False):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)

        x4 = torch.nn.functional.leaky_relu(self.bn1(self.mid_conv1(x3)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn2(self.mid_conv2(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn3(self.mid_conv3(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.mid_conv4(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn5(self.mid_conv5(x4)), 0.2)

        out = self.up_block2(x2, x4)
        out = self.up_block3(x1, out)
        out = torch.nn.functional.relu(self.last_bn(self.last_conv1(out)))
        out = self.last_conv2(out)
        # change the output
        out = (out + x)
        # out = torch.sigmoid(out) #Try tanh and scale
        # out = torch.nn.functional.relu(self.last_bn2(self.last_conv2(out)))
        # out = torch.nn.functional.relu(self.last_conv2(out))
        # out = self.linear1(out.reshape(self.opts.batch_size, -1))
        # out = self.linear2(out).reshape(self.opts.batch_size, 1, 64, 520)
        # out = torch.nn.functional.sigmoid(out)
        # out = self.softplus(out)
        out = torch.sigmoid(out)
        return out


        
class Sino_repair_net():
    
    def __init__(self, opts, device_ids, load_model=False):
        self.opts = opts
        self.epoch_step = 0
        self.model_num = 0
        # self.network = UNet(opts)
        # self.network = test2.Net(opts)
        self.network = VGG.vgg19_bn()
        # self.network = framelet.Framelets()
        # import VGG
        # self.network = VGG.vgg19_bn()
        # self.network = Nets.AlexNet(channelnumber=1, num_classes=64*520)

        
        #Logic to make training on a GPU cluster easier
        if torch.cuda.device_count() > 1 and opts.max_gpus > 1:
            if len(device_ids) <= opts.max_gpus:
                self.network = torch.nn.DataParallel(self.network) #, device_ids=device_ids[0]
            else:
                self.network = torch.nn.DataParallel(self.network, device_ids=device_ids[0:opts.max_gpus-1])
        self.network.cuda()
        
        # Create two sets of loss functions
        self.loss_func_l1 = torch.nn.L1Loss()
        self.loss_func_MSE = torch.nn.MSELoss()
        self.mssim_loss = MSSSIM(window_size=11, size_average=True)

        self.loss_func_poss = torch.nn.PoissonNLLLoss()
        self.loss_func_KLDiv = torch.nn.KLDivLoss()
        self.loss_func_Smoothl1 = torch.nn.SmoothL1Loss()
        self.loss_func_part = torch.nn.L1Loss()
        self.test_loss = torch.nn.MSELoss(reduction='none')

        self.OPT_count = 0
        #TODO2: Change the load model dict
        if self.opts.load_model == True or load_model:
            print("Restoring model")
            try:
                if load_model:
                    self.network.load_state_dict(torch.load('/home/liang/Desktop/output/model/model_dict_0'))

                else:
                    self.network.load_state_dict(torch.load(os.path.join(self.opts.output_path, 'model', 'model_dict')))
            except:
                # original saved file with DataParallel
                state_dict = torch.load(os.path.join(self.opts.output_path, 'model', 'model_dict'))
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # name = k[7:] # remove `module.`
                    name = 'module.'+ k
                    new_state_dict[name] = v
                # load params
                self.network.load_state_dict(new_state_dict)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img, valid=None):
        if not valid:
            output = self.network.forward(input_img)
            loss, loss2 = self.optimize(output, target_img)
            return output, loss, loss2
        else:

            final = input_img.clone()
            mask = torch.tensor([i for i, n in enumerate(valid) if n==1]).cuda()
            print(mask)
            if len(mask)>0:
                traininput = torch.index_select(input_img, 0, mask)
                trainoutput = self.network.forward(traininput)
                final[mask] = trainoutput
                loss, loss2 = self.optimize(final, target_img)
            else:
                loss, loss2 = self.loss_func_l1(final, target_img), (1 - self.mssim_loss.forward(final, target_img))/2

            return final, loss, loss2

    def test(self, x, y, valid=None):
        if not valid:
            output = self.network.forward(x)
            loss = self.test_loss(output, y).detach()
            return output, loss
        else:
            final = x.clone()
            mask = torch.tensor([i for i, n in enumerate(valid) if n == 1]).cuda()
            # print(mask)
            if len(mask) > 0:
                traininput = torch.index_select(x, 0, mask)
                trainoutput = self.network.forward(traininput)#, test=True)#.detach()
                final[mask] = trainoutput
            loss = self.test_loss(final, y).detach()


        return final, loss

    def optimize(self, output, target_img):
        #TODO: can add other loss terms if needed
        #TODO: need to step though this code to make sure it works correctly

        # # Including l1 loss
        # mask = ((input_img * -1.0) + 1.0) >= 0.8
        loss1 = self.loss_func_l1(output, target_img) #+ self.loss_func_MSE(output, target_img)
        l1 = loss1.detach()

        # # # Including a consistency loss
        loss2 = (1 - self.mssim_loss.forward(output, target_img))/2
        l2 = loss2.detach()

        # loss3 = self.loss_func_MSE(output, target_img)
        #
        # if self.OPT_count == 0:
        #     self.alpha = torch.tensor(0.5).cuda()
        #     self.lossl1 = []
        #     self.lossmssim = []
        #
        # self.lossl1.append(l1.item())
        # self.lossmssim.append(l2.item())
        #
        # if self.OPT_count >= 20:
        #     self.alpha = torch.FloatTensor(self.lossl1).mean().cuda()/(torch.FloatTensor(self.lossl1).mean() + torch.FloatTensor(self.lossmssim).mean()).cuda()
        #     self.OPT_count = 0
        #
        # # loss = self.alpha * loss1+(1-self.alpha) *loss2
        # vx = output - torch.mean(output)
        # vy = target_img - torch.mean(target_img)
        # loss_pearson_correlation = 1 - torch.sum(vx * vy) / (
        #             torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2)))  # use Pearson correlation

        loss = loss1 #+ loss2
        # loss = loss1s
        # loss = loss + filter_loss
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.OPT_count += 1

        return l1, l2

    def save_network(self):
        print("saving network parameters")
        folder_path = os.path.join(self.opts.output_path, 'model')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.network.state_dict(), os.path.join(folder_path, "model_dict_{}".format(self.model_num)))
        self.model_num += 1
        if self.model_num >= 5: self.model_num = 0


if __name__ == '__main__':
    opts = options.parse()
    # net = UNet(opts).cuda()
    # Change loading module for single test
    net = Sino_repair_net(opts, [0, 1], load_model=True)
    net.cuda()
    # print(net)

    import matplotlib.pylab as plt
    import numpy as np


    def plot_img(img):
        # Pass in the index to read one of the sinogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Original (Sinogram)")
        ax.set_xlabel("Projection position (pixels)")
        ax.set_ylabel("Projection angle (deg)")
        img = np.transpose(img)
        ax.imshow(img)

    from datasets import Sino_Dataset
    # data = Sino_Dataset('/home/liang/Desktop/dataset.pkl', 5, testing=False, input_depth=3)
    data = Sino_Dataset(opts.test_file, 5, testing=True, input_depth=opts.input_channel_number, output_depth=opts.output_channel_number)

    output = []
    for i, (good, bad) in enumerate(data):
        # out = good
        # if i == 0:
        #     for j in range(len(out)):
        #         output.append(out[j])
        # else:
        #     output.append(out[-1])
            # print(i)
        # print(good.shape)
        # bad = bad[0]
        # good = good[0]

        good /= (good.max() + 1e-8)
        bad /= (bad.max() + 1e-8)

        test_x = torch.from_numpy(bad)
        test_x = test_x.unsqueeze(0)
        # # test_x = test_x.unsqueeze(0)
        test_x = Variable(test_x).cuda()
        # #
        test_y = torch.from_numpy(good)
        test_y = test_y.unsqueeze(0)
        # # test_y = test_y.unsqueeze(0)
        test_y = Variable(test_y).cuda()

        # out_x = net.network.forward(test_x)
        # out_x, loss = net.test(test_y, test_x)
        # out_x = out_x.squeeze()
        # out_x = out_x.cpu()
        # const = out_x.detach().numpy()
        # plot_img(good* 255)
        # plot_img(bad* 255)
        # plot_img(const* 255)
        # # print(loss)
        #
        # break
    output = np.stack(output)
    print(output.shape)
    plt.show()


    #
    # test_x = Variable(torch.FloatTensor(1, 1, 64, 520)).cuda()
    # out_x = net.forward(test_x)


