import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from dataset import MaskBaseDataset, kfold
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def ensembling(models,num_classes,device):
    if len(models)==2:
        print('ENSEMBLING..{0},{1}'.format(models[0],models[1]))
        model_module1 = getattr(import_module("model"), models[0])
        model_module2 = getattr(import_module("model"), models[1])
        
        model1 = model_module1(
            num_classes=num_classes,
        ).to(device)

        model2 = model_module2(
            num_classes=num_classes,
        ).to(device)
        
        model_module=getattr(import_module("ensemble"), 'myensemble2')

        model=model_module(
            modelA=model1,
            modelB=model2,
            num_classes=num_classes
        ).to(device)

    elif len(models)==3:
        print('ENSEMBLING..{0},{1},{2}'.format(models[0],models[1],models[2]))
        model_module1 = getattr(import_module("model"), models[0])
        model_module2 = getattr(import_module("model"), models[1])
        model_module3 = getattr(import_module("model"), models[2])
        
        model1 = model_module1(
            num_classes=num_classes,
        ).to(device)

        model2 = model_module2(
            num_classes=num_classes,
        ).to(device)

        model3 = model_module3(
            num_classes=num_classes,
        ).to(device)
        model_module=getattr(import_module("ensemble"), 'myensemble3')

        model=model_module(
            modelA=model1,
            modelB=model2,
            modelC=model3,
            num_classes=num_classes
        ).to(device)
    elif len(models)==4:
        print('ENSEMBLING..{0},{1},{2},{3}'.format(models[0],models[1],models[2],models[3]))
        model_module1 = getattr(import_module("model"), models[0])
        model_module2 = getattr(import_module("model"), models[1])
        model_module3 = getattr(import_module("model"), models[2])
        model_module4 = getattr(import_module("model"), models[3])
        model1 = model_module1(
            num_classes=num_classes,
        ).to(device)

        model2 = model_module2(
            num_classes=num_classes,
        ).to(device)

        model3 = model_module3(
            num_classes=num_classes,
        ).to(device)
        model4 = model_module4(
            num_classes=num_classes,
        ).to(device)
        model_module=getattr(import_module("ensemble"), 'myensemble4')

        model=model_module(
            modelA=model1,
            modelB=model2,
            modelC=model3,
            modelD=model4,
            num_classes=num_classes
        ).to(device)
    return model

def set_augment(data_subset_, mode, dataset):
    transform_module = getattr(import_module("dataset"), mode) # args.augment_train)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    data_subset_.dataset.set_transform(transform) 
    return data_subset_



def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18
    
    
    best_val_acc = 0
    best_val_loss = np.inf
    fold_avg_acc = 0
    fold_avg_loss = 0
    data_set_list = dataset.split_dataset()
    

    for fold,train_set,val_set in data_set_list:
        if 'kfold' in args.dataset:
            print(f'FOLD {fold}')
            print('--------------------------------')
        # Define data loaders for training and testing data in this fold
        train_set = set_augment(train_set, args.augment_train, dataset)
        val_set = set_augment(val_set, args.augment_valid, dataset)
        train_loader = torch.utils.data.DataLoader(
                        train_set, 
                        batch_size=args.batch_size,
                        num_workers=multiprocessing.cpu_count() // 2,
                        pin_memory=use_cuda,
                        drop_last=True,)
        val_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=args.valid_batch_size,
                        num_workers=multiprocessing.cpu_count() // 2,
                        shuffle=False,
                        pin_memory=use_cuda,
                        drop_last=True)
        
        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        
        if "vit" in args.model:
            print(max(args.resize))
            model = model_module(
                num_classes=num_classes,
                image_size=max(args.resize)
            ).to(device)
        else:
            model = model_module(
                num_classes=num_classes
            ).to(device)

        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)



        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_acc=max(best_val_acc,val_acc)
                
                import earlystopping
                earlystop=earlystopping.EarlyStopping(patience=args.patience,delta=args.delta)
                best_val_loss,cnt=earlystop(best_val_loss,val_loss,model,save_dir,cnt)
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
                if earlystop.early_stop:
                    print('EARLY STOPPED!')
                    break
        fold_avg_acc += val_acc
        fold_avg_loss += val_loss
    fold_avg_acc /= (fold+1)
    fold_avg_loss /= (fold+1)
    print(
        f"fold avg acc : {fold_avg_acc:4.2%}, fold avg loss: {fold_avg_loss:4.2} || "
        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
    print()


                

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augment_train', type=str, default='CustomAugm_train', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--augment_valid', type=str, default='CustomAugm_val', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[512, 384], help='resize size for image when training')   
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)') 
    parser.add_argument('--valid_batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='resnet18', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp_0228_a', help='model save at {SM_MODEL_DIR}/{name}')  # Tensorboard에 저장되는 이름
    parser.add_argument('--ensemble', nargs="+", type=str, default=0,help="ensemble model names")
    parser.add_argument('--patience', type=int, default=5,help="early_stopping patience")
    parser.add_argument('--delta', type=float, default=0,help="early_stopping delta")

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
