import config_c2f as config
from dataset_c2f import open_set_folds, face_dataset, select_p_images_per_class,ijbc_dataset, partition_dataset
from models import fetch_encoder as fetch_encoder
from  models import head as head
from finetune_c2f import fine_tune, weight_gallery_base
from utils_c2f import save_dir_far_curve, save_dir_far_excel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import pprint
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import datetime


# for boolean parser argument
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("'True' or 'False' expected")

def False_or_float(v):
    if v == "False":
        return False
    else:
        return float(v)

parser = argparse.ArgumentParser()
# basic arguments
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--lr",default=1e-3,type=float)
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--num_epochs",default=1,type=int,help="num_epochs for fine-tuning")

# dataset arguments
parser.add_argument("--dataset", type=str, default='CASIA', help="['CASIA','IJBC']")
parser.add_argument("--probe_dataset", type=str, default='probe', help="['probe','val']")
parser.add_argument("--num_gallery", type=int, default=3, help="number of gallery images per identity")

# encoder arguments
parser.add_argument("--encoder", type=str, default='VGG19', help="['VGG19','Res50']")
parser.add_argument("--head_type", type=str, default='cdpl', help="[ 'cos', 'cdpl']")

# main arguments: classifier init / finetune layers / matcher
parser.add_argument("--classifier_init", type=str, default='MSRWC',help="weight calibration")
parser.add_argument("--finetune_layers", type=str, default='BN',help="['None','Full','Partial','PA','BN']")  # 'None' refers to no fine-tuning
parser.add_argument("--matcher", type=str, default='NAC',help="['NAC','cos']")

# misc. arguments: no need to change
parser.add_argument("--cos_s",default=32,type=float,help="scale for CosFace")
parser.add_argument("--cos_m",default=0.4,type=float,help="margin for CosFace")
parser.add_argument("--train_output",type=str2bool,default=False,help="if True, train output layer")
parser.add_argument("--k",default=16,type=int,help="k for NAC")

# MSRWC arguments
parser.add_argument("--p",default=4,type=int,help="p for vggface")
parser.add_argument("--base_weight",default=0.1,type=float)
parser.add_argument("--top_k",default=8749,type=int)

# CDPL arguments
parser.add_argument("--a",default=0.1,type=float,help="a for loss")
parser.add_argument("--t",default=0.7,type=float,help="t for loss")
parser.add_argument("--r",default=0.1,type=float,help="r for loss")
args = parser.parse_args()



def main(args):
    # check arguments
    global classifier
    assert args.finetune_layers in ['None','Full','Partial','PA','BN'], \
        "finetune_layers must be one of ['None','Full','Partial','PA','BN']"

    # fix random seed
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # set device
    args.device = torch.device(args.device_id)

    # save result
    if args.finetune_layers == 'None':
        exp_name = 'Pretrained'
    else:
        exp_name = f'{args.classifier_init}_{args.finetune_layers}/{args.head_type}'

    current_time = datetime.datetime.now()
    if args.head_type == "cdpl":
        interval_name = f'a={args.a}_t={args.t}_t={args.t}_r={args.r}'
    else:
        interval_name = ''
    # Format the time as a string, for example: 2022-01-01_12-30-45
    time_str = current_time.strftime("%Y-%m-%d__%H:%M:%S")
    save_dir = f"results/{args.dataset}_{args.encoder}_{args.probe_dataset}/{exp_name}/{interval_name}/{time_str}/"
    os.makedirs(save_dir, exist_ok=True)
    print("results are saved at: ", save_dir)

    # save arguments
    argdict = args.__dict__.copy()
    argdict['device'] = argdict['device'].type + f":{argdict['device'].index}"
    with open(save_dir + '/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)


    train_trf = transforms.Compose([
        transforms.RandomResizedCrop(size=112, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    eval_trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])

    base_trf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])

    """
    prepare G, K, U sets for evaluation
    increase batch_size for faster inference
    """

    # prepare dataset: config
    data_config = config.data_config[args.dataset]
    if args.dataset == 'CASIA':
        folds = open_set_folds(data_config["image_directory"], data_config["known_list_path"],
                               data_config["unknown_list_path"], args.num_gallery)
        dataset_val = face_dataset(folds.val, eval_trf, img_size=112)
        dataset_gallery = face_dataset(folds.G, eval_trf, img_size=112)
        dataset_probe = face_dataset(folds.test, eval_trf, img_size=112)
        data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
        data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)
        data_loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=4)
        num_cls = folds.num_known
        trainset_gallery = face_dataset(folds.G, train_trf, 112)

    if args.dataset == 'IJBC':
        Gallery, Known, Unknown, Probe, Val, Test, num_cls = partition_dataset(data_config["ijbc_t_m"],
                                                                               data_config["ijbc_5pts"],
                                                                               data_config["ijbc_gallery_1"],
                                                                               data_config["ijbc_gallery_2"],
                                                                               data_config["ijbc_probe"],
                                                                               data_config["image_directory"],
                                                                               data_config["plk_file_root"],
                                                                               args.num_gallery)
        dataset_gallery = ijbc_dataset(Gallery, eval_trf, img_size=112)
        data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
        dataset_probe = ijbc_dataset(Test, eval_trf, img_size=112)
        data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)
        dataset_val = ijbc_dataset(Val, eval_trf, img_size=112)
        data_loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=4)
        trainset_gallery = ijbc_dataset(Gallery, train_trf, 112)
    base_dataset = datasets.ImageFolder('/home/chen/python_workspace/face/OSFI-by-FineTuning-main/vggface2_mtcnn_160', transform=base_trf)
    base_subset = select_p_images_per_class(base_dataset, args.p)
    data_loader_base = DataLoader(base_subset, batch_size=256, shuffle=False, num_workers=4)
    num_cls_base = len(base_dataset.classes)

    '''
    prepare encoder
    '''
    encoder = fetch_encoder.fetch(args.device, config.encoder_config,
                                  args.encoder, args.finetune_layers, args.train_output)

    '''
    fine-tune
    '''
    if args.finetune_layers != "None":  # for 'None', no fine-tuning is done
        if args.head_type == "cos":
            classifier = head.cosface_head(512, num_cls, s=args.cos_s, m=args.cos_m)
        elif args.head_type == "cdpl":
            classifier = head.cdpl_head(512, num_cls, s=args.cos_s, m=args.cos_m, a=args.a, t=args.t, r=args.r)
        classifier.to(args.device)

        # classifier initialization
        if args.classifier_init == 'MSRWC':
            prototypes = weight_gallery_base(args, encoder, data_loader_gallery, num_cls, data_loader_base,num_cls_base, 512)
            classifier.weight = nn.Parameter(prototypes.T)

        else:
            pass  # just use random weights for classifier

        # set optimizer & LR scheduler
        optimizer = optim.Adam([{"params": encoder.parameters(), "lr": args.lr},
                                {"params": classifier.parameters(), "lr": args.lr}],
                               weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)


    train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)
    trainset_base = datasets.ImageFolder(
        '/home/chen/python_workspace/face/OSFI-by-FineTuning-main/vggface2_mtcnn_160', transform=train_trf)
    trainset_base_subset=select_p_images_per_class(trainset_base, args.p)
    train_loader_base = DataLoader(trainset_base_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    fine_tune(args, train_loader_gallery,num_cls, train_loader_base,num_cls_base,encoder, classifier, optimizer, scheduler, verbose=True)

    '''
    evaluate encoder
    '''
    encoder.eval()
    flip = transforms.RandomHorizontalFlip(p=1)
    Gfeat = torch.FloatTensor([]).to(args.device)
    Glabel = torch.LongTensor([])
    for img, label in tqdm(data_loader_gallery):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Gfeat = torch.cat((Gfeat, feat), dim=0)
        Glabel = torch.cat((Glabel, label), dim=0)
    Pfeat = torch.FloatTensor([]).to(args.device)
    Plabel = torch.LongTensor([])
    if args.probe_dataset == "probe":
        data_loader = data_loader_probe
    if args.probe_dataset == "val":
        data_loader = data_loader_val
    for img, label in tqdm(data_loader):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Pfeat = torch.cat((Pfeat, feat), dim=0)
        Plabel = torch.cat((Plabel, label), dim=0)
    Gfeat = Gfeat.cpu()
    Pfeat = Pfeat.cpu()
    Gprototypes = weight_gallery_base(args, encoder, data_loader_gallery, num_cls, data_loader_base, num_cls_base, 512)
    Gprototypes = Gprototypes.cpu()

    # save results
    save_dir_far_curve(args, Gprototypes, Glabel, Pfeat, Plabel, save_dir)
    save_dir_far_excel(args, Gprototypes, Glabel, Pfeat, Plabel, save_dir)


if __name__ == '__main__':
    pprint.pprint(vars(args))
    main(args)
