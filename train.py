import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from model import *
from multi_read_data import MemoryFriendlyLoader


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=3000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='Rebuttal/000/', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
csv_path = args.save+'/csv_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'train.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    model = Network(stage=args.stage)

    model = model.cuda()
    optimizer_a = torch.optim.Adam(model.ha.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    optimizer_b = torch.optim.Adam(list(model.hb.parameters()) + list(model.calibrate.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)


    train_low_data_names = 'Your training data path'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = 'Your testing data path'
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        num_workers=0, shuffle=False, generator=torch.Generator(device = 'cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        num_workers=0, shuffle=False, generator=torch.Generator(device = 'cuda'))

    total_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            

            total_step += 1
            input = Variable(input, requires_grad=True).cuda()

            _, loss1 , loss2 , loss3  = model._loss_Jiaoti(input)
            
            if total_step %10 < 7:
                for param in model.ha.parameters():
                    param.requires_grad = True
                for param in model.hb.parameters():
                    param.requires_grad = False
                for param in model.calibrate.parameters():
                    param.requires_grad = False
                optimizer_a.zero_grad()

                loss = loss1
                loss.backward()
                optimizer_a.step()
            else:
                for param in model.ha.parameters():
                    param.requires_grad = False
                for param in model.hb.parameters():
                    param.requires_grad = True
                for param in model.calibrate.parameters():
                    param.requires_grad = True
                optimizer_b.zero_grad()

                loss = (loss2 + loss3) 
                loss.backward()
                optimizer_b.step()
            
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            losses.append(loss.item())
            logging.info('train-epoch-{:0>3d}-step-{:0>5d}  Tloss:{:<8.4f} loss1:{:<8.4f} loss2:{:<8.4f} loss3:{:<8.4f}'\
                         .format(epoch, batch_idx, loss,loss1,loss2,loss3))

            if total_step % 500 == 0 and total_step != 0:
                logging.info('train %03d %f', epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_queue):
                        input = Variable(input, volatile=True).cuda()
                        image_name = image_name[0].split('/')[-1].split('.')[0]
                        _, ref_list, _, _= model(input)
                        for ii in range(1):
                            u_name = '{}_{}_{}.png'.format(image_name, total_step, ii)
                            u_path = image_path + '/' + u_name
                            save_images(ref_list[ii], u_path)
                model.train()
                logging.info('train-epoch %03d %f', epoch, np.average(losses))
                utils.save(model, os.path.join(model_path, 'weights_%d_%d.pt' % (epoch, total_step)))



if __name__ == '__main__':
    main()
