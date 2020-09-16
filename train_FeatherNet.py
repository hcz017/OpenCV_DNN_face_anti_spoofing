import os
import cv2
import argparse
import time
import yaml
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
# sklearn libs
from sklearn.metrics import confusion_matrix
import pickle
import train.models as models
from read_data import Data

model_names = sorted(name for name in models.__dict__
                  if name.islower() and not name.startswith("__")
                  and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                	choices=model_names,
                	help='models architecture: ' +
                     	' | '.join(model_names) +
                     	' (default: resnet18)')
parser.add_argument('--config', default='train/cfgs/FeatherNet.yaml')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                	help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                	help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                	help='manual epoch number (useful on restarts)')
parser.add_argument("--random-seed", type=int, default=14,
                    	help='Seed to provide (near-)reproducibility.')
parser.add_argument('--gpus', type=str, default='0', help='use gpus training eg.--gups 0,1')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                	metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                	metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                	help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                	metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                	metavar='N', help='print frequency (default: 10)')
parser.add_argument('--val', '--evaluate', dest='evaluate', default=False, type=bool,
                	help='evaluate models on validation set')
parser.add_argument('--normalization', default=False, type=bool,
                	help='whether to normalize train data')
parser.add_argument('--phase-test', default=False, type=bool,
                    help='whether testing in test dataset ')
parser.add_argument('--center-crop', default=False, type=bool,
                	help='whether to do center crop')
parser.add_argument('--input_size', default=224, type=int, help='img crop size')
parser.add_argument('--image_size', default=224, type=int, help='ori img size')
parser.add_argument('--model_name', default='', type=str, help='name of the models')
parser.add_argument('--every-decay', default=40, type=int, help='how many epoch decay the lr')
parser.add_argument('--fl-gamma', default=3, type=int, help='gamma for Focal Loss')

best_prec1 = 0
model_names = sorted(name for name in models.__dict__
                 	if name.islower() and not name.startswith("__")
                 	and callable(models.__dict__[name]))

USE_GPU = torch.cuda.is_available()

def main():
    global args, best_prec1, USE_GPU,device
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    ## Set random seeds ##
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create models
    if "model" in config.keys():
        model = models.__dict__[args.arch](**config['model'])
    else:
        model = models.__dict__[args.arch]()
        device = torch.device('cuda:' + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
        str_input_size = '1x3x224x224'
    if USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        model = torch.nn.DataParallel(model,device_ids=args.gpus)
        model.to(device)

    # define loss function (criterion) and optimizer
    criterion = FocalLoss(device,2,gamma=args.fl_gamma)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Data loading code
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],std=[0.10050353,    0.100842826, 0.10034215])
    img_size = args.input_size
    ratio = 224.0 / float(img_size)
    train_trans = [transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
    if args.normalization == True:
        train_trans.append(normalize)

    val_trans = []
    if args.center_crop == True:
        val_trans.append(transforms.Resize(int(256 * ratio)))
        val_trans.append(transforms.CenterCrop(img_size))
    else:
        val_trans.append(transforms.Resize((img_size, img_size)))
    val_trans.append(transforms.ToTensor())
    if args.normalization == True:
        val_trans.append(normalize)
    
    train_dataset = Data(transforms.Compose(train_trans), phase_train=True)
    val_dataset = Data(transforms.Compose(val_trans), phase_train=False,phase_test=args.phase_test)
    
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion,args.start_epoch)
        return
    
    print("[INFO] start training…")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            print('epoch: {} The best is {} last best is {}'.format(epoch,prec1,best_prec1))
        best_prec1 = max(prec1, best_prec1)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        save_model_dir = '{}/{}_{}_{}_best.pth'.format(args.save_path, time_stp, args.model_name, epoch) if is_best else'{}/{}_{}_{}.pth'.format(args.save_path, time_stp, args.model_name, epoch)
        torch.save(model, save_model_dir)

        # save_onnx_dir = '{}/{}_{}_{}_best.onnx'.format(args.save_path, time_stp, args.model_name, epoch) if is_best else'{}/{}_{}_{}.onnx'.format(args.save_path, time_stp, args.model_name, epoch)
        # dummy_input = torch.randn(1, 3, 224, 224)
        # torch.onnx.export(model, dummy_input, save_onnx_dir, verbose=True)

        save_name = '{}/{}_{}_{}_best.pth.tar'.format(args.save_path, time_stp, args.model_name, epoch) if is_best else'{}/{}_{}_{}.pth.tar'.format(args.save_path, time_stp, args.model_name, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)

def train(train_loader, model, criterion, optimizer, epoch):
      batch_time = AverageMeter()
      data_time = AverageMeter()
      losses = AverageMeter()
      top1 = AverageMeter()

      # switch to train mode
      model.train()
      end = time.time()

      for i, (input, target) in enumerate(train_loader):
          # measure data loading time
          data_time.update(time.time() - end)
          input_var = Variable(input).float().to(device)
          target_var = Variable(target).long().to(device)

          # compute output
          output = model(input_var)
          loss = criterion(output, target_var)
          prec1,prec2 = accuracy(output.data, target_var,topk=(1,2))

          # measure accuracy and record loss
          reduced_prec1 = prec1.clone()

          top1.update(reduced_prec1[0])

          reduced_loss = loss.data.clone()
          losses.update(reduced_loss)

          # compute gradient and do SGD step
          optimizer.zero_grad()
          loss.backward()
          #  check whether the network is well connected
          optimizer.step()

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()
          lr = optimizer.param_groups[0]['lr']

          if i % args.print_freq == 0:
              line = 'Epoch: [{0}][{1}/{2}]\t lr:{3:.5f}\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                      .format(epoch, i, len(train_loader),lr,
                      batch_time=batch_time, loss=losses, top1=top1)
              print(line)

def validate(val_loader, model, criterion,epoch):
    global time_stp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time() 
    with torch.no_grad():
        for i, (input, target,depth_dirs) in enumerate(val_loader):
            with torch.no_grad():
                input_var = Variable(input).float().to(device)
                target_var = Variable(target).long().to(device)

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1,prec2 = accuracy(output.data, target_var,topk=(1,2))
                losses.update(loss.data, input.size(0))
                top1.update(prec1[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    line = 'Test: [{0}/{1}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1)
                    print(line)
    return top1.avg
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.every_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, device,class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha =   Variable(torch.ones(class_num, 1)) # 0.25 *
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.tensor(alpha))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
       
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss 
if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    main()
 
