import argparse

parser = argparse.ArgumentParser(
    description='Train and Test on ADNI Dataset')
# datasets config 
parser.add_argument('--file_dir' , default="DATA_PATH",type=str,help="learning rate")

# training config 
# parser.add_argument('--gpus', nargs='+', type=int)
parser.add_argument('--gpus' , default=1,type=int,help="gpu_id")
parser.add_argument('--lr' , default=1e-4,type=float,help="learning rate")
parser.add_argument('--batchSize' , default=1,type=int,help="batch size")
parser.add_argument('--epoch' , default=500,type=int,help="train iter epoch")
parser.add_argument('--epoch_start' , default=0,type=int,help="start from epoch")
parser.add_argument('--cont_training' , default=False,type=bool,help="cont_training")

# model config
parser.add_argument('--model_name',default="DC2Fusion",type=str)
parser.add_argument('--inputSize' , default=(160, 192, 224),type=list,help="inputSize")
parser.add_argument('--fineSize' , default=(160, 192, 224),type=list,help="list")
parser.add_argument('--embed_dim' , default=48,type=int,help="embed_dim")


# loss config
parser.add_argument('--alpha1' , default=2,type=float,help="mri alpha loss ")
parser.add_argument('--alpha2' , default=1,type=float,help="pet alpha loss ")
parser.add_argument('--beta1' , default=2,type=float,help="mri beat loss")
parser.add_argument('--beta2' , default=1,type=float,help="pet beat loss")
parser.add_argument('--gamma1' , default=2,type=float,help="mri beat loss")
parser.add_argument('--gamma2' , default=1,type=float,help="pet beat loss")
config = parser.parse_args()