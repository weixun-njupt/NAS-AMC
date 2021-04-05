import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from thop import profile
from thop import clever_format
from torchsummary import summary

#from torchstat import stat

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='twocell', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 8

TAcc=open('/data/wx/PC-DARTS/wx/logs2/train_acc.txt',mode='w')
Tloss=open("/data/wx/PC-DARTS/wx/logs2/train_loss.txt",mode="w")
VAcc=open('/data/wx/PC-DARTS/wx/logs2/vaild_acc.txt',mode='w')
Vloss=open("/data/wx/PC-DARTS/wx/logs2/vaild_loss.txt",mode="w")
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  #print(genotype)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  #stat(model, (1, 16, 16))


  model = model.cuda()
 # input_size=(1,2,128)
  '''
  Input = torch.randn(1, 1, 2, 128)
  Input = Input.type(torch.cuda.FloatTensor)
  macs, params = profile(model, inputs=(Input,), custom_ops={Network: Network.forward})
  macs, params = clever_format([macs, params], "%.3f")
  print(macs, params)
  summary(model,(1,16,16))
'''

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  '''
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
  '''

  x = loadmat("/data/wx/PC-DARTS/data/datashujudoppler=100.mat")
  x = x.get('train_data')
  x = np.reshape(x, [-1, 1, 2, 128])
  data_num = 22000
  y1 = np.zeros([data_num, 1])
  y2 = np.ones([data_num, 1])
  y3 = np.ones([data_num, 1]) * 2
  y4 = np.ones([data_num, 1]) * 3
  y5 = np.ones([data_num, 1]) * 4
  y6 = np.ones([data_num, 1]) * 5
  y7 = np.ones([data_num, 1]) * 6
  y8 = np.ones([data_num, 1]) * 7
  y = np.vstack((y1, y2, y3, y4, y5, y6, y7, y8))
  y = np.array(y)
  X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=30)

  X_train = torch.from_numpy(X_train)
  Y_train = torch.from_numpy(Y_train)
  X_train = X_train.type(torch.FloatTensor)
  Y_train = Y_train.type(torch.LongTensor)

  Y_train = Y_train.type(torch.LongTensor)
  # Y_train=np.reshape(Y_train,(16800,4))
  Y_train = Y_train.squeeze()
  print(Y_train.type)
  print(Y_train)
  print(X_train.shape, Y_train.shape)
  train_Queue = torch.utils.data.TensorDataset(X_train, Y_train)
  print(train_Queue)
  X_val = torch.from_numpy(X_val)
  Y_val = torch.from_numpy(Y_val)
  X_val = X_val.type(torch.FloatTensor)
  Y_val = Y_val.type(torch.LongTensor)
  # Y_train = one_hot_embedding(Y_train, 4)
  Y_val = Y_val.type(torch.LongTensor)
  Y_val = Y_val.squeeze()
  print(Y_val.type, Y_val)
  valid_Queue = torch.utils.data.TensorDataset(X_val, Y_val)


  train_queue = torch.utils.data.DataLoader(train_Queue, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                            num_workers=2)

  valid_queue = torch.utils.data.DataLoader(valid_Queue, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  writer = SummaryWriter('./logs/balei')
  '''
  Input = torch.randn(1, 1, 2, 128)
  Input=Input.type(torch.cuda.FloatTensor)
  macs, params = profile(model, inputs=(Input,),custom_ops={Network:Network.forward})
  macs, params = clever_format([macs, params], "%.3f")
  print(macs, params)
  '''
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    writer.add_scalar('Train/acc', train_acc, epoch)
    TAcc.write(str(train_acc)+",")
    writer.add_scalar('Train/loss', train_obj, epoch)
    Tloss.write(str(train_obj)+",")

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
     
    if valid_acc > best_acc:
        best_acc = valid_acc
        utils.save(model, os.path.join(args.save, 'weights_best_acc.pt'))
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
    writer.add_scalar('Valid/acc', valid_acc, epoch)
    VAcc.write(str(valid_acc)+",")
    writer.add_scalar('Valid/loss', valid_obj, epoch)
    Vloss.write(str(valid_obj)+",")
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    #loss=(loss-0.169).abs()+0.169
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 4))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 4))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  main()
  TAcc.close()
  Tloss.close()
  VAcc.close()
  Vloss.close()
