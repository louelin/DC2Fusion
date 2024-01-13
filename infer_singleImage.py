
import os

import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
from monai import transforms
from models.DC2Fusion import DC2Fusion 
# from models.DXconvMorpher import CONFIGS as CONFIGS
from config_diff import config

import warnings
from evaluation_metrics import psnr , ssim , nmi , mutual_information , fsim


warnings.filterwarnings('ignore')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '9'
# torch.cuda.set_device(5)

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

# set_seed(3407)

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
    
    '''
    Initialize model
    '''
    
    model = DC2Fusion(n_channels=1 , embed_dim = 96 , window_size = (2,2,2)).cuda()

    
    pth_path = "...pth"
    model.load_state_dict(torch.load(pth_path).state_dict())

    '''
    Initialize training
    '''
    val_composed = transforms.Compose([transforms.CropForegroundd(keys=("mri","pet"),source_key="mri",select_fn=threshold_at_one, margin=0),
                                        transforms.SpatialPadd(keys=("mri","pet"),spatial_size=(224,224,224),method='symmetric', mode='constant') ,
                                        transforms.Resized(keys=("mri","pet"),spatial_size=(128,128,128)),
                                        transforms.ScaleIntensityd(keys=("mri","pet")),
                                       ])

    

    val_subjects = []
    with open("/home/fanxx/fxx/Fusion/Fusion_MRI_PET/data/test.csv" , "r") as f :
        val_subjects = f.readline().replace("\n","").split(",")
    val_set = datasets.dataset(file_path="/home/fanxx/fxx/Fusion/Fusion_MRI_PET/preproce_data2" , subject_names= val_subjects , transform= val_composed)

    
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    
    psnrs, ssims, nmis, mis, fsims = [], [], [], [], []

    with torch.no_grad():
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

            print("psnr_mri : {:.3f} , ssim_mri : {:.3f} , nmis_mri : {:.3f} , mis_mri : {:.3f} , fsims_mri : {:.3f}".format(
                psnr_val1,ssim_val1,nmi_val1,mi_val1,fsim_val1
            ))

            print("psnr_pet : {:.3f} , ssim_pet : {:.3f} , nmis_pet : {:.3f} , mis_pet : {:.3f} , fsims_pet : {:.3f}".format(
                psnr_val2,ssim_val2,nmi_val2,mi_val2,fsim_val2
            ))
            print("--------------")
        
        print("mean psnr : {:.3f} +- {:.3f} , mean ssim : {:.3f} +- {:.3f}, mean nmis : {:.3f} +- {:.3f}, mean mis : {:.3f} +- {:.3f}, mean fsims : {:.3f} +- {:.3f}".format(
                np.mean(psnrs),np.std(psnrs), np.mean(ssims),np.std(ssims), np.mean(nmis),np.std(nmis), np.mean(mis),np.std(mis),np.mean(fsims),np.std(fsims)
            ))



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    torch.cuda.set_device("cuda:" + str(config.gpus))
    main()
# --coding:utf-8--