import model
import torch
import time
import imageio
from utils import *
import numpy as np

class Test(object):
    def __init__(self, model_path, save_path,kernel, scale, conf, method_num, num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        self.max_iters=num_of_adaptation
        self.display_iter = 1

        self.upscale_method= 'cubic'
        self.noise_level = 0.0

        self.back_projection=False
        self.back_projection_iters=4

        self.model_path=model_path
        self.save_path=save_path
        self.method_num=method_num

        self.ds_method=methods[self.method_num]

        self.kernel = kernel
        self.scale=scale
        self.scale_factors = [self.scale, self.scale]

        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.psnr=[]
        self.iter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.Net()

        self.learning_rate = 0.02
        self.loss_fn = torch.nn.L1Loss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)

    def __call__(self, img, gt, img_name):
        self.img = img
        self.gt = modcrop(gt, self.scale)

        self.img_name = img_name
        print('** Start Adaptation for X', self.scale, os.path.basename(self.img_name), ' **')

        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.sf = np.array(self.scale_factors)
        self.output_shape = np.uint(np.ceil(np.array(self.img.shape[0:2]) * self.scale))

        print('[*] Baseline ')
        self.train()


    def train(self):
        self.hr_father = self.img
        self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
        self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)

        t1=time.time()

        if self.iter==0:
            self.learning_rate=2e-2
        elif self.iter < 4:
            self.learning_rate=1e-2
        else:
            self.learning_rate=5e-3
        
        self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father)
        # Display information
        if self.iter % self.display_iter == 0:
                print('Scale: ', self.scale, ', iteration: ', (self.iter+1), ', loss: ', self.loss[self.iter])

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))

    def quick_test(self):
        # 1. True MSE
        self.sr = self.forward_pass(self.img, self.gt.shape)

        self.mse = self.mse + [np.mean((self.gt - self.sr)**2)]

        '''Shave'''
        scale = int(self.scale)
        PSNR = psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])
        self.psnr.append(PSNR)
        
        # 2. Reconstruction MSE
        self.reconstruct_output = self.forward_pass(self.hr2lr(self.img), self.img.shape)
        self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))

        processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)

        print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)

        return processed_output

    def forward_pass(self, input, output_shape=None):
        ILR = imresize(input, self.scale, output_shape, self.upscale_method)
        
        output = self.model(ILR)

        return np.clip(np.squeeze(output), 0., 1.)

    def forward_backward_pass(self, input, hr_father):
        ILR = imresize(input, self.scale, hr_father.shape, self.upscale_method)

        HR = hr_father[None, :, :, :]

        train_output = self.model(ILR)
        self.loss = self.loss_fn(train_output, HR)
        
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

        return np.clip(np.squeeze(train_output), 0., 1.)
    
    def hr2lr(self, hr):
        lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
        return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)

    def final_test(self):
    
        output = self.forward_pass(self.img, self.gt.shape)
        if self.back_projection == True:
            for bp_iter in range(self.back_projection_iters):
                output = back_projection(output, self.img, down_kernel=self.kernel,
                                                  up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)

        processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)

        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  rgb2y(processed_output)[scale:-scale, scale:-scale])

        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
        #           rgb2y(processed_output))

        self.psnr.append(PSNR)

        return processed_output