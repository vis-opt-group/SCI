import os
import sys
import numpy as np
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--steps', type=float, default=100, help='finetune steps')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--save', type=str, default='results/finetune/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/difficult.pt', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.makedirs(args.save, exist_ok=True)

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

    model = Finetunemodel(args.model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    train_low_data_names = './data/finetune'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')
    test_low_data_names = './data/finetune'
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True)

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)

    total_step = 0

    model.train()
    for step in range(args.steps):

        for batch_idx, (input, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()

            optimizer.zero_grad()

            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            print('finetune-step:{} loss:{}'.format(step, loss.item()))

            if total_step % 10 == 0 and total_step != 0:
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_queue):
                        input = Variable(input).cuda()
                        image_name = image_name[0].split('\\')[-1].split('.')[0]
                        illu, ref = model(input)

                        u_name = '%s.png' % (image_name + '_' + str(total_step) + '_ref_')
                        u_path = args.save + '/' + u_name
                        save_images(ref, u_path)


if __name__ == '__main__':
    main()
