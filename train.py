"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import shutil
from argparse import ArgumentParser
import onnx
import onnxruntime as ort

# import os
# os.environ['LOCAL_RANK'] = '0'
# local_rank = os.environ.get('LOCAL_RANK', '0')

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model_baseline import SSD, ResNet 
from src.utils import generate_dboxes, Encoder, coco_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate
from src.dataset import collate_fn, CocoDataset
import yaml


def read_yaml(config_path):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' does not exist.")
    except yaml.YAMLError as exc:
        print(f"Error while reading '{config_path}': {exc}")

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco",
                        help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default="trained_models",
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD")

    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
                        help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--rt", type=str, default="val",help="'Onnxruntime','val'")
    parser.add_argument("--epochs", type=int, default=1, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=1, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")
    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)

    # parser.add_argument('--local_rank', default=0, type=int,
    #                     help='Used for multi-process training. Can either be manually set ' +
    #                          'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument("-c","--config_path", type=str, required=True, help="Path to yaml configuration")
    parser.add_argument("--mtype", default="fp32_model", choices=["fp16_model","fp32_model","QDQ_model","evaluate_tvm","fp16_MP"])
    parser.add_argument("--deploy_cfg", default="config/tvm_cpu.py", type=str, help="deploy config path")
    parser.add_argument("--subset", default=10, type=int, help="Number of images to use from each folder")



    args = parser.parse_args()
    return args


def main(opt):
    print("inside main has reached")
    # if torch.cuda.is_available():
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #     num_gpus = torch.distributed.get_world_size()
    #     torch.cuda.manual_seed(123)
    # else:
    #     torch.manual_seed(123)
    #     num_gpus = 1
    num_gpus=1
    # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    if opt.model == "ssd":
        dboxes = generate_dboxes(model="ssd")
        model = SSD(backbone=ResNet(), num_classes=len(coco_classes))
    else:
        dboxes = generate_dboxes(model="ssdlite")
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    checkpoint_path = "/kaggle/input/ssd/pytorch/res_50/1/SSD.pth"

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])

    first_epoch = 0    
    if opt.rt == "Onnxruntime":
        print("ONNX runtime")
    if opt.rt == "val":
        print("baseline validation")
    print("evaluation part has reached")
    for epoch in range(first_epoch, opt.epochs):
        evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold,opt.mtype,opt.rt,config,opt.deploy_cfg,opt.subset)


if __name__ == "__main__":
    opt = get_args()
    config = read_yaml(opt.config_path)
    # env_dist = os.environ
    # args.local_rank=int(env_dist['LOCAL_RANK'])
    main(opt)
