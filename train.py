import argparse

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.CIL

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch.nn as nn
import numpy as np
import torch

import numpy as np
import time

import torch
import torch.nn as nn


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, k ,known_class , all_class ,fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)
        
        weight_m = fc

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]
        # print(CV_temp.shape)torch.Size([32, 512, 512])
        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))
        # print(sigma2.shape)torch.Size([32, 20, 20])
        sigma2 = sigma2.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        
        # print(sig)
        aug_result = y + 0.5 * sigma2 #  * k

        return aug_result

    def forward(self , k , known_class , all_class ,features, y ,fc , target_x, ratio):

        self.estimator.update_CV(features.detach(), target_x)

        isda_aug_y = self.isda_aug(k , known_class , all_class ,fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

        return isda_aug_y 

def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.start_class != -1:
        cfg.DATASET.START_CLASS = args.task_id * 10

    if args.end_class != -1:
        cfg.DATASET.END_CLASS = (args.task_id + 1) *10
    
    if args.task_id != -1:
        cfg.DATASET.TASK_ID = args.task_id

    if args.task_num != -1:
        cfg.DATASET.TASK_NUM = args.task_num


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.CIL = CN()
    cfg.TRAINER.CIL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CIL.CSC = False  # class-specific context
    cfg.TRAINER.CIL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CIL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CIL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    # 将输入的参数加到 cfg 中
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    

    print_args(args, cfg)
    
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    start_class = cfg.DATASET.START_CLASS
    end_class = cfg.DATASET.END_CLASS
    trainer = build_trainer(cfg)
        

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    # 切换训练模型
    if not args.no_train:
        # trainer.train()
        trainer.max_epoch = 10
        start_class = 0
        end_class = 20
        for i in range(10):
            if i != 0:
                start_class = end_class 
                end_class = end_class + 20
            trainer.task_id = i

            temp , k = trainer.data_manager.get_dataset(np.arange(0, end_class),source="train", mode="train")
            classnames = trainer.get_classname(temp)

            k = (10 - i) / 10

            trainer.model.prompt_learner = trainers.CIL.Prompt(cfg, classnames, trainer.clip_model , trainer.device)
              

            from dassl.optim import build_optimizer, build_lr_scheduler
            from torch.cuda.amp import GradScaler
            trainer.model.to(trainer.device)
            trainer.optim = build_optimizer(trainer.model, cfg.OPTIM)
            trainer.sched = build_lr_scheduler(trainer.optim, cfg.OPTIM)
            trainer.register_model("prompt_learner", trainer.model, trainer.optim, trainer.sched)

            trainer.scaler = GradScaler() if cfg.TRAINER.CIL.PREC == "amp" else None
            

            isda_criterion = ISDALoss(int(512), end_class).cuda()
            
            trainer.incremen tal_train(start_class , end_class  , k = k,criterion = isda_criterion , mode = "train")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--task_id", type=int, default=-1
    )
    parser.add_argument(
        "--task_num", type=int, default=-1
    )
    parser.add_argument(
        "--start_class", type=int, default=0
    )
    parser.add_argument(
        "--end_class", type=int, default=10
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
