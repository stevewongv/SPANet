import os
import sys
import cv2
import argparse
import numpy as np
import logging

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import skimage.measure as ms
import progressbar

from dataset import TrainValDataset, TestDataset
from cal_ssim import SSIM
from SPANet import SPANet


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
torch.cuda.manual_seed_all(2019)
torch.manual_seed(2019)



def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.device = torch.device("cuda")
        
        self.log_dir = './logdir'
        self.model_dir = './model'
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        self.log_name = 'train_derain'
        self.val_log_name = 'val_derain'
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.test_data_path = 'testing/real_test_1000.txt'                 # test dataset txt file path
        self.train_data_path = 'training/real_world.txt'          # train dataset txt file path
        
        self.multi_gpu = True
        self.net = SPANet().to(self.device)
        self.l1 = nn.L1Loss().to(self.device)
        self.l2 = nn.MSELoss().to(self.device)
        self.ssim = SSIM().to(self.device)
        
        self.step = 0
        self.save_steps = 400
        self.num_workers = 16
        self.batch_size = 16
        self.writers = {}
        self.dataloaders = {}
        self.shuffle = True
        self.opt = Adam(self.net.parameters(), lr=5e-3)
        self.sche = MultiStepLR(self.opt, milestones=[30000], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name, train_mode=True):
        dataset = {
            True: TrainValDataset,
            False: TestDataset,
        }[train_mode](dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
        if train_mode:
            return iter(self.dataloaders[dataset_name])
        else:
            return self.dataloaders[dataset_name]

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name,mode='train'):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            self.net.load_state_dict({k.replace('module.',''):v for k,v in obj['net'].items()})
        except FileNotFoundError:
            return

        if mode == 'train':
            self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step
    
        
    def inf_batch(self, name, batch):
        if name == 'test':
            torch.set_grad_enabled(False)
        O, B,M = batch['O'], batch['B'],batch['M']
        O, B,M = O.to(self.device), B.to(self.device),M.to(self.device)

        mask,out = self.net(O)
        
        if name == 'test':
            return out.cpu().data , batch['B'],O,mask
        
        # loss
        l1_loss = self.l1(out,B)
        mask_loss = self.l2(mask[:,0,:,:],M)
        ssim_loss = self.ssim(out,B)
        
        loss = l1_loss  +  (1-ssim_loss) + mask_loss
        
        # log
        losses = {
            'l1_loss' : l1_loss.item() 
        }
        l2 = {
            'mask_loss' : mask_loss.item() 
        }
        losses.update(l2)
        ssimes = {
            'ssim_loss' : ssim_loss.item() 
        }        
        losses.update(ssimes)
        allloss = {
            'all_loss' : loss.item() 
        }
        losses.update(allloss)
        return  out,mask,M, loss, losses


    def heatmap(self,img):
        if len(img.shape) == 3:
            b,h,w = img.shape
            heat = np.zeros((b,3,h,w)).astype('uint8')
            for i in range(b):
                heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
        else:
            b,c,h,w = img.shape
            heat = np.zeros((b,3,h,w)).astype('uint8')
            for i in range(b):
                heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
        return heat

    def save_mask(self, name, img_lists,m = 0):
        data, pred, label,mask,mask_label = img_lists
        pred = pred.cpu().data
        
        mask = mask.cpu().data
        mask_label = mask_label.cpu().data
        data, label,pred,mask,mask_label = data * 255, label * 255, pred * 255, mask*255,mask_label*255
        pred = np.clip(pred, 0, 255)
        
        mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
        mask_label = np.clip(mask_label.numpy(), 0, 255).astype('uint8')
        h, w = pred.shape[-2:]
        mask = self.heatmap(mask)
        mask_label = self.heatmap(mask_label)
        gen_num = (1, 1)

        img = np.zeros((gen_num[0] * h, gen_num[1] * 5 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx],mask[idx],mask_label[idx]]
                    for k in range(5):
                        col = (j * 5 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.png' % (self.step, name))
        cv2.imwrite(img_file, img)



def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    if sess.multi_gpu :
        sess.net = nn.DataParallel(sess.net)
    sess.tensorboard(sess.log_name)
    sess.tensorboard(sess.val_log_name)

    dt_train = sess.get_dataloader(sess.train_data_path)
    dt_val = sess.get_dataloader(sess.train_data_path)

    while sess.step < 40001:
        sess.sche.step()
        sess.net.train()
        sess.net.zero_grad()

        batch_t = next(dt_train)
        pred_t,mask_t,M_t, loss_t, losses_t = sess.inf_batch(sess.log_name, batch_t)
        sess.write(sess.log_name, losses_t)
        loss_t.backward()
        sess.opt.step() 

        if sess.step % 4 == 0:
            sess.net.eval()
            batch_v = next(dt_val)
            pred_v,mask_v,M_v, loss_v, losses_v = sess.inf_batch(sess.val_log_name, batch_v)
            sess.write(sess.val_log_name, losses_v)
        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_mask(sess.log_name, [batch_t['O'], pred_t, batch_t['B'],mask_t,M_t])
            if sess.step % 4 == 0:
                sess.save_mask('valderain5', [batch_v['O'], pred_v, batch_v['B'],mask_v,M_v])
            logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name,'test')
    if sess.multi_gpu :
        sess.net = nn.DataParallel(sess.net)
    sess.batch_size = 1
    sess.shuffle = False
    sess.outs = -1
    dt = sess.get_dataloader(sess.test_data_path, train_mode=False)

    ssim = []
    psnr = []

    widgets = [progressbar.Percentage(),progressbar.Bar(),progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=len(dt)).start()
    for i, batch in enumerate(dt):
        pred, B, losses,mask = sess.inf_batch('test', batch)
        pred,B =pred[0], B[0]
        mask = mask.cpu().data
        mask = mask*255
        mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
        mask = sess.heatmap(mask)
        mask = np.transpose(mask[0], (1, 2, 0))
        pred = np.transpose(pred.numpy(), (1, 2, 0))
        B = np.transpose(B.numpy(), (1, 2, 0))
        pred = np.clip(pred, 0, 1)
        B = np.clip(B, 0, 1)    
        ssim.append(ms.compare_ssim(pred,B,multichannel=True))
        psnr.append(ms.compare_psnr(pred,B))
        pred = pred * 255
        ensure_dir('../realtest/derain5_real/')
        cv2.imwrite('../realtest/derain5_real/{}.png'.format(i+1),pred)
        cv2.imwrite('../realtest/derain5_real/{}m.jpg'.format(i+1),mask)
        bar.update(i+1)
    print(np.mean(ssim),np.mean(psnr))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='test')
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

