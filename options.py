import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description="Training S3MDHR...")

# Input Parameters
parser.add_argument('--savename',type=str,help='Saving files name', default='SMDHR_b')
parser.add_argument("--root",type=str,required=False,help="Root of HSI data", default="G:\datasets\HSI-MD",)# "G:\datasets\HSI-MD" "G:\datasets\HSI-MD\Chikusei\Chikusei"
parser.add_argument("--in_channel",type=int,required=False,help="Number of input HSIs channels", default=305,)
parser.add_argument("--batchsize", type=int, required=False, help="Batch size", default=8)
parser.add_argument("--lr", type=float, required=False, help="Learning rate", default=2e-4)  # 3
parser.add_argument("--epochs", type=int, required=False, help="Maximum epochs", default=300)
parser.add_argument("--imgsize",type=int,required=False,help="Image size for inference (val)", default=256,)
parser.add_argument("--emb_dim", type=int, required=False, help="Embedding dimension", default=64)
parser.add_argument("--n_tf",type=int,required=False,help="Number of transformer blocks of enhacement block", default=2,)
parser.add_argument("--n_layers",type=int,nargs="+",required=False,help="Number of layers in a decoder", default=[2, 1],)
parser.add_argument("--n_head",type=int,required=False,help="Number of attention heads for all transformer-based blocks", default=8,)
parser.add_argument("--win_size",nargs="+",type=int,required=False,help="Window size for all swin-transformer-based blocks", default=[8,8,8],)
parser.add_argument("--pat_size",nargs="+",type=int,required=False,help="Patch size for all swin-transformer-based blocks", default=[4,4,4],)
parser.add_argument("--randcrop", type=str2bool, required=False, help="Random crop", default=True)
parser.add_argument("--cropsize", type=int, required=False, help="Crop size", default=32)
parser.add_argument("--long",type=str2bool,required=False,help="is the prompt long?", default=False)
parser.add_argument("--intp",type=str2bool,help="is interpolate?", default=True)
parser.add_argument("--state_dict",type=str,required=False,help="Restored model with relative/absolute path", default=None,) #"D:\colacomo\code\PromptHSI-main\ckpt\Chikusei\BEST-Ablation1_bc.pth""I:\PromptHSI-main\ckpt\EP300-PromptHSI.pth BEST-OneRestore_b.pth"
parser.add_argument("--complex_loss",type=str,required=False,help="Restored model with complex loss function", default=True)
# for inference
parser.add_argument("--ckpt",type=str,required=False,help="Pretrained model with relative/absolute path", default='ckpt/EP300-SMDHR81_b.pth',)

options = parser.parse_args()

