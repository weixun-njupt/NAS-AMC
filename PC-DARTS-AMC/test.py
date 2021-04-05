import os
import sys
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
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import time
from thop import profile
from thop import clever_format
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import itertools
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='eval-twocell-20201121-181244/weights_best_acc.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 8

mods = ['BPSK', 'QPSK', '8PSK', '16QAM', '2FSK', '4FSK', '8FSK', 'MSK']
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
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  total = 0
  for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
      total += m.weight.data.shape[0]

  bn = torch.zeros(total)
  index = 0
  for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
      size = m.weight.data.shape[0]
      bn[index:(index + size)] = m.weight.data.abs().clone()
      index += size

  y, i = torch.sort(bn)
  thre_index = int(total * 0.5)
  thre = y[thre_index]

  pruned = 0
  cfg = []
  cfg_mask = []
  for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
      weight_copy = m.weight.data.clone()
      mask = weight_copy.abs().gt(thre).float().cuda()
      pruned = pruned + mask.shape[0] - torch.sum(mask)
      m.weight.data.mul_(mask)
      m.bias.data.mul_(mask)
      cfg.append(int(torch.sum(mask)))
      cfg_mask.append(mask.clone())
      print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
      cfg.append('M')

  pruned_ratio = pruned / total

  print('Pre-processing Successful!')
  utils.load(model, args.model_path)
  
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()


  snr=[0,2,4,6,8,10,12,14,16,18,20]
  sum=0
  TIM=0
  utils.load(model, args.model_path)
  for i in snr:
    data_num = 1000
    test_data = loadmat("/data/wx/PC-DARTS/data/testdoppler=100/snr="+str(i)+".mat")
    x = test_data.get("train_data")
    print(x.shape)
    x = np.reshape(x, [-1, 1, 2, 128])
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
    X_test = torch.from_numpy(x)
    Y_test = torch.from_numpy(y)
    X_test = X_test.type(torch.FloatTensor)
    Y_test = Y_test.type(torch.LongTensor)
    Y_test = Y_test.squeeze()


    test_Queue = torch.utils.data.TensorDataset(X_test, Y_test)

    test_queue = torch.utils.data.DataLoader(
      test_Queue, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
   
    if i ==0:
      Input = torch.randn(1, 1, 2, 128)
      Input = Input.type(torch.cuda.FloatTensor)
      macs, params = profile(model, inputs=(Input,))
      macs, params = clever_format([macs, params], "%.3f")
      print("flops    params")
      print(macs, params)
      summary(model,input_size=(1,2,128))


    time1 = time.time()
    test_acc, test_obj ,target,loggg= infer(test_queue, model, criterion)

    time2 = time.time() - time1
    logging.info('第 %d snr test_acc %f', i,test_acc)
    sum+=test_acc
    logging.info("第 %d snr time: %f", i, time2)
    TIM+=time2


    #print(target)
    #print(target.shape)
    #print(loggg.shape)
    target = target.cpu().detach().numpy()
    loggg = loggg.cpu().detach().numpy()
    cm = confusion_matrix(target, loggg)
    #print(cm)



    
    #plot_confusion_matrix(cm ,mods, title=" Confusion Matrix ( SNR=%d dB)" % (i))
    '''
    if i>=10:
      address_jpeg = '/data/wx/PC-DARTS/wx/picture/' + '100' + 'hz-CSS-snr=' + str(i) + '.pdf'
      plt.savefig(address_jpeg)
    plt.close('all') '''
    #plt.show()

    ACC=sum/11
    TT=TIM/11
    print("average acc : ", ACC) 
    print("average time : ", TT)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    logits, _ = model(input)

    #print(logits.shape)
    if step==0:
      FTA=target
      probabily = torch.nn.functional.softmax(logits, dim=1)
      max_value, index = torch.max(probabily, 1)
      FTL=index
      loss = criterion(logits, target)

      # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 4))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    else:
      FTA=torch.cat((FTA,target),dim=0)

      probabily = torch.nn.functional.softmax(logits, dim=1)
      max_value,index=torch.max(probabily,1)
      FTL = torch.cat((FTL, index), dim=0)
      #class_index=result_(index)
     # print(index)
      loss = criterion(logits, target)

        # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 4))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg,FTA,FTL

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  ###????

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)  # rotation=45 x?????45?
  plt.yticks(tick_marks, classes)
  plt.tight_layout(pad=2)

  ###????
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
       plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
  ###????

  '''
  iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
  # # ij??????????
  iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (cm.size, 2))
  for i, j in iters :
      plt.text(j, i, format(cm[i, j]))  # ???????
  '''
  plt.ylabel('True label')
  plt.xlabel('Predicted label')



if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  main()

