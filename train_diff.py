from pickletools import optimize
from torch.utils.tensorboard import SummaryWriter
import os
import utils

import glob
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
import torch.nn as nn
from monai import transforms
import matplotlib.pyplot as plt
from natsort import natsorted

from models.DC2Fusion import DC2Fusion 


from config_diff import config
from torch.optim import Adam

import datetime
import warnings

from evaluation_metrics import psnr , ssim , nmi , mutual_information , fsim

import monai 


warnings.filterwarnings('ignore')



def threshold_at_one(x):
    # threshold at 1
    return x > 0


### 超参数=0.6
def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(3407)


#用conda创建一个新环境
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():

    save_dir = "{}_lr_[{}]_loss_[{:.1f}]_[{:.1f}]_[{:.1f}]_[{:.1f}]_[{:.1f}]_[{:.1f}]".format(config.model_name , config.lr , config.alpha1, config.alpha2
                                                            ,config.beta1,config.beta2 , config.gamma1, config.gamma2)
    print(save_dir)
    if not os.path.exists('result/' + save_dir):
        os.makedirs('result/' + save_dir)
    if not os.path.exists('result/' + save_dir + "/log"):
        os.makedirs('result/' + save_dir + "/log")
    # sys.stdout = Logger('result/' + save_dir + "/log")
    epoch_start = config.epoch_start
    cont_training = config.cont_training
    max_epoch = config.epoch

    time1 = datetime.datetime.now()
    print(time1)
    # 打印按指定格式排版的时间
    time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_txt = 'result/' + save_dir + "/log" + time2 + ".txt"
    with open(log_txt , "a") as f :
        for k,v in sorted(vars(config).items()):
            print(k,'=',v , file=f)

    '''
    Initialize model
    '''
    
    # config = CONFIGS['DXConv-Morph']
    model = DC2Fusion(n_channels=1 , embed_dim = 96 , window_size = (2,2,2)).cuda()

    
    # pth_path = "..pth"
    # model.load_state_dict(torch.load(pth_path).state_dict())

    
    '''
        If continue from previous training
    '''
    if cont_training:

        model_dir = 'result/' + save_dir
        updated_lr = round(config.lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        list_names  = natsorted(os.listdir(model_dir))
        pth_names = [name for name in list_names if ".pth" in name]
        print(model_dir + "/" + pth_names[-1])
        best_model = torch.load(model_dir + "/" + pth_names[-1])
        model.load_state_dict(best_model)
        print("succeessfully load pth")
        
    else:
        updated_lr = config.lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
        transforms.CropForegroundd(keys=("mri","pet"),source_key="mri",select_fn=threshold_at_one, margin=0),
        transforms.SpatialPadd(keys=("mri","pet"),spatial_size=(224,224,224),method='symmetric', mode='constant') ,
        transforms.Resized(keys=("mri","pet"),spatial_size=(128,128,128)),
        transforms.RandRotated(keys=("mri","pet"),range_x=[0.2,0.2], prob=0.1, keep_size=True),
        transforms.RandRotated(keys=("mri","pet"),range_y=[0.2,0.2], prob=0.1, keep_size=True),
        transforms.RandRotated(keys=("mri","pet"),range_z=[0.2,0.2], prob=0.1, keep_size=True),
        transforms.RandAdjustContrastd(keys=("mri"),prob=0.5, gamma=(0.1, 2)), 
        transforms.RandGaussianNoised(keys=("mri"),prob=0.5,mean=0.5,std=0.1),
        # RandHistogramShift(prob=0.2 , )
        transforms.ScaleIntensityd(keys=("mri","pet")),
    ])

    val_composed = transforms.Compose([transforms.CropForegroundd(keys=("mri","pet"),source_key="mri",select_fn=threshold_at_one, margin=0),
                                        transforms.SpatialPadd(keys=("mri","pet"),spatial_size=(224,224,224),method='symmetric', mode='constant') ,
                                        transforms.Resized(keys=("mri","pet"),spatial_size=(128,128,128)),
                                        transforms.ScaleIntensityd(keys=("mri","pet")),
                                       ])

    with open(log_txt , "a") as f :
        print("-----------train transforms---------------" , file = f)
        for trans1 in train_composed.transforms:
            print(trans1 , file = f)
        print("-----------val transforms ----------------" , file = f)
        for trans1 in val_composed.transforms:
            print(trans1 , file = f)


    train_subjects = [] 
    with open("/home/fanxx/fxx/Fusion/Fusion_MRI_PET/data/train.csv" , "r") as f :
        train_subjects = f.readline().replace("\n","").split(",")
    train_set = datasets.dataset(file_path="/home/fanxx/fxx/Fusion/Fusion_MRI_PET/preproce_data2" , subject_names= train_subjects , transform= train_composed)

    val_subjects = []
    with open("/home/fanxx/fxx/Fusion/Fusion_MRI_PET/data/test.csv" , "r") as f :
        val_subjects = f.readline().replace("\n","").split(",")
    val_set = datasets.dataset(file_path="/home/fanxx/fxx/Fusion/Fusion_MRI_PET/preproce_data2" , subject_names= val_subjects , transform= val_composed)

    train_loader = DataLoader(train_set, batch_size=config.batchSize,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    # Set optimizer
    opt = Adam(model.parameters(),lr=updated_lr)
    # opt = SGD(model.parameters(), lr=updated_lr, momentum=0.9)
    lncc_loss_fn = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3)
    ssim_loss_fn = monai.losses.ssim_loss.SSIMLoss(spatial_dims=3)

    # pixel-level loss function
    l1_loss_fn = nn.L1Loss()

    # evaluate metrics

    # Jacobian_loss = utils.jacobian_determinant_vxm #雅可比行列式
    writer = SummaryWriter(log_dir='result/' + save_dir + "/log")

    

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        for i, data in enumerate(train_loader):
            model.train()
            adjust_learning_rate(opt, epoch, max_epoch, config.lr)
            data = [t.cuda() for t in data]
            mri = data[0].float()
            pet = data[1].float()
           

            # optimizer
            opt.zero_grad()
            # df, warp, warp_y, w_label_m_to_f, w_label_f_to_m = model(data)
            fusion_Image = model(mri,pet)

            # struct-level loss
            lncc_loss = [lncc_loss_fn(fusion_Image , mri) , lncc_loss_fn(fusion_Image , pet)]
            ssim_loss = [ssim_loss_fn(fusion_Image , mri) , ssim_loss_fn(fusion_Image , pet)]
            
            # pixel-level loss
            l1_loss = [l1_loss_fn(fusion_Image , mri) , l1_loss_fn(fusion_Image,pet)]
            

            loss = config.alpha1 * lncc_loss[0] + config.alpha2 * lncc_loss[1] + config.beta1 * ssim_loss[0] + config.beta2 * ssim_loss[1] + config.gamma1 * l1_loss[0] + config.gamma2 * l1_loss[1]
            loss_all.update(loss.item())
           
            log_str = "[{:.1f}]_[{:.1f}]_[{:.1f}]_[{:.1f}]_ : epoch: {}  i : {} loss : {:.5f} lncc : {:.5f} ssim : {:.5f} , l1 : {:.5f}".format(
                config.alpha1 ,config.alpha2 , config.beta1 , config.beta2 , epoch , i , loss.item() , 
                sum(lncc_loss).item(), sum(ssim_loss).item(),sum(l1_loss).item())
            print(log_str)
            with open(log_txt, 'a') as f:
                f.write(log_str + '\n')

            opt.zero_grad()
            loss.backward()
            opt.step()
            # break

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        writer.add_graph(model,[d.float() for d in data])
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         writer.add_histogram(name + '/grad', param.grad, epoch)

        '''
        Validation
        '''
        

        with torch.no_grad():
            psnrs, ssims, nmis, mis, fsims = [], [], [], [], []
            for n, data in enumerate(val_loader):
                model.eval()
                data = [t.cuda() for t in data]
                mri = data[0].float()
                pet = data[1].float()

                # flow, val_warp_x, val_warp_y,w_label_m_to_f,_ = model(data)
                fusion_Image = model(mri,pet)

                gt1 = mri[0,0,:,:,64].detach().cpu()
                gt2 = pet[0,0,:,:,64].detach().cpu()
                final = fusion_Image[0,0,:,:,64].detach().cpu()
                psnr_val1 = psnr(final, gt1)
                psnr_val2 = psnr(final, gt2)
                psnr_val = (psnr_val1 + psnr_val2) / 2
                psnrs.append(psnr_val)

                ssim_val1 = ssim(final.unsqueeze(0).unsqueeze(0), gt1.unsqueeze(0).unsqueeze(0))
                ssim_val2 = ssim(final.unsqueeze(0).unsqueeze(0), gt2.unsqueeze(0).unsqueeze(0))
                ssim_val = (ssim_val1 + ssim_val2) / 2
                ssims.append(ssim_val)

                nmi_val1 = nmi(final, gt1)
                nmi_val2 = nmi(final, gt2)
                nmi_val = (nmi_val1 + nmi_val2) / 2
                nmis.append(nmi_val)

                mi_val1 = mutual_information(final, gt1)
                mi_val2 = mutual_information(final, gt2)
                mi_val = (mi_val1 + mi_val2) / 2
                mis.append(mi_val)

                fsim_val1 = fsim(final, gt1)
                fsim_val2 = fsim(final, gt2)
                fsim_val = (fsim_val1 + fsim_val2) / 2
                fsims.append(fsim_val)
                
                log_str = "psnr_mri : {:.3f} , ssim_mri : {:.3f} , nmis_mri : {:.3f} , mis_mri : {:.3f} , fsims_mri : {:.3f}".format(
                psnr_val1,ssim_val1,nmi_val1,mi_val1,fsim_val1)
                print(log_str)
                with open(log_txt, 'a') as f:
                    f.write(log_str + '\n')

                log_str ="psnr_pet : {:.3f} , ssim_pet : {:.3f} , nmis_pet : {:.3f} , mis_pet : {:.3f} , fsims_pet : {:.3f}".format(
                psnr_val2,ssim_val2,nmi_val2,mi_val2,fsim_val2)
                print(log_str)
                with open(log_txt, 'a') as f:
                    f.write(log_str + '\n')

                log_str = "------------------"
                print(log_str)
                with open(log_txt, 'a') as f:
                    f.write(log_str + '\n')
        log_str = "mean psnr : {:.3f} +- {:.3f} , mean ssim : {:.3f} +- {:.3f}, mean nmis : {:.3f} +- {:.3f}, mean mis : {:.3f} +- {:.3f}, mean fsims : {:.3f} +- {:.3f}".format(
                np.mean(psnrs),np.std(psnrs), np.mean(ssims),np.std(ssims), np.mean(nmis),np.std(nmis), np.mean(mis),np.std(mis),np.mean(fsims),np.std(fsims)
            )
        print(log_str)
        with open(log_txt, 'a') as f:
            f.write(log_str + '\n')

        eval_metrics = 0.1 * np.mean(psnrs) +  np.mean(ssims) + np.mean(nmis) + np.mean(mis) + np.mean(fsims)
        log_str = "eval_metrics : {} ".format(eval_metrics)
        print(log_str)
        with open(log_txt, 'a') as f:
            f.write(log_str + '\n')
        
        torch.save(model.state_dict(), 'result/' + save_dir + '/' +
                   'dsc{:.3f}.pth'.format(eval_metrics))
        writer.add_scalar('psnr/validate', np.mean(psnrs), epoch)
        writer.add_scalar('ssim/validate', np.mean(ssims) , epoch)
        writer.add_scalar('nmi/validate', np.mean(nmis), epoch)
        writer.add_scalar('mi/validate', np.mean(mis), epoch)
        writer.add_scalar('fsim/validate', np.std(fsims), epoch)
        plt.switch_backend('agg')
        mri_image = comput_fig(data[0])
        pet_image = comput_fig(data[1])
        fusion_image = comput_fig_jet(fusion_Image)
        overlap_image = comput_fig_overlap(data[0],data[1])
        
        writer.add_figure('mri', mri_image, epoch)
        plt.close(mri_image)
        writer.add_figure('pet', pet_image, epoch)
        plt.close(pet_image)
        writer.add_figure('fusion', fusion_image, epoch)
        plt.close(fusion_image)
        writer.add_figure('fusion_GT', overlap_image, epoch)
        plt.close(overlap_image)
        

    writer.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def comput_fig_jet(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='jet')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def comput_fig_overlap(img1,img2):
    img1 = img1.detach().cpu().numpy()[0, 0, 48:64, :, :]
    img2 = img2.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img1.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img1[i, :, :], cmap='gray')
        plt.imshow(img2[i, :, :], cmap='jet',alpha=0.4)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    torch.cuda.set_device("cuda:" + str(config.gpus))
    main()
# --coding:utf-8--
