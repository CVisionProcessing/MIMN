import torch
from torchvision import transforms

import os
import sys
import time
import random
import numpy as np

import torch.nn.functional as F

from backbone_SK import Encoder, Decoder

from args import get_parser
from utils.utils import get_optimizer
from utils.utils import make_dir, check_parallel
from dataset import TrainDataset, ValDataset
from utils.utils import save_checkpoint_epoch, load_checkpoint_epoch
from measures.jaccard import db_eval_iou_multi
import train_loss

torch.cuda.empty_cache()

def trainIters(args):
    print(args)

    model_dir = os.path.join('ckpt/', args.model_name)
    make_dir(model_dir)

    encoder = Encoder()
    decoder = Decoder()

    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    print('encoder_params:{}, decoder_params:{}'.format(len(encoder_params),len(decoder_params)))
    para_encoder = sum([np.prod(list(p.size())) for p in encoder.parameters()])
    para_decoder = sum([np.prod(list(p.size())) for p in decoder.parameters()])
    print('Model {} : params: {:4f}M ; Model {} : params: {:4f}M '.format(encoder._get_name(), para_encoder * 4 / 1000 / 1000, decoder._get_name(), para_decoder * 4 / 1000 / 1000))
    
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                            args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                            args.weight_decay_cnn)

    # Training Data
    trainDatasets = ['DUTS-TR','DAVIS']
    dataset = TrainDataset([args.root + d for d in trainDatasets], args.imsize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset), len(dataloader)))

    # val data
    valDatasets = ['DAVIS']
    valDataset = ValDataset([args.root + d for d in valDatasets], args.imsize)
    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    print("Val Set, DataSet Size:{}, DataLoader Size:{}".format(len(valDataset), len(valDataloader)))
    
    best_iou = 0

    start = time.time()
    
    for e in range(args.max_epoch):
        
        batch_idx= 0
        epoch_losses = {'loss': [], 'iou': []}
        val_losses = {'valLoss': [], 'valIou': []}
        print("Epoch", e)

        encoder.train(True)
        decoder.train(True)

        for data in dataloader:
            batch_idx += 1
            image, flow, label = data['video'].cuda(), data['fwflow'].cuda(), data['label'].cuda()
            B, C, H, W = image.size()

            spatial, spatial_feature, temporal, temporal_feature = encoder(image, flow)
            out = decoder(spatial, spatial_feature, temporal, temporal_feature)
            
            loss = train_loss.bce_ssim_loss(out, label)
            
            iou = db_eval_iou_multi(label.cpu().detach().numpy(), out.cpu().detach().numpy())

            dec_opt.zero_grad()
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            dec_opt.step()

            epoch_losses['loss'].append(loss.data.item())
            epoch_losses['iou'].append(iou)
            
            if (batch_idx + 1) % args.print_every == 0:
                mt = np.mean(epoch_losses['loss'])
                miou = np.mean(epoch_losses['iou'])

                te = time.time() - start
                print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                        '\tIOU: {:.4f}'.format(e, args.max_epoch, batch_idx,
                                                len(dataloader), te, mt, miou))
                with open(os.path.join('ckpt', args.model_name, 'loss.txt'),'a') as f:
                    f.write('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                        '\tIOU: {:.4f}\n'.format(e, args.max_epoch, batch_idx,
                                                len(dataloader), te, mt, miou))
                f.close()
                start = time.time()
        if (e + 1) % 3 == 0:
            with torch.no_grad():
                for data in valDataloader:
                    image, flow, label = data['video'].cuda(), data['fwflow'].cuda(), data['label'].cuda()

                    B, C, H, W = image.size()

                    spatial, spatial_feature, temporal, temporal_feature = encoder(image, flow)
                    out = decoder(spatial, spatial_feature, temporal, temporal_feature)

                    loss = train_loss.bce_ssim_loss(out, label)
                    iou = db_eval_iou_multi(label.cpu().detach().numpy(), out.cpu().detach().numpy())
                    val_losses['valLoss'].append(loss.data.item())
                    val_losses['valIou'].append(iou)
                    
                mt = np.mean(val_losses['valLoss'])
                miou = np.mean(val_losses['valIou'])
                print('Epoch: [{}/{}]\tValLoss: {:.4f}\tvalIou: {:.4f}'.format(e, args.max_epoch, mt, miou))
                with open(os.path.join('ckpt', args.model_name, 'loss.txt'),'a') as f:
                    f.write('Epoch: [{}/{}]\tValLoss: {:.4f}\tvalIou: {:.4f}\n'.format(e, args.max_epoch, mt, miou))
                f.close()
                
            if miou > best_iou:
                best_iou = miou
                save_checkpoint_epoch(args, encoder, decoder,
                                    enc_opt, dec_opt, e, False)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print('gpu_id: ', args.gpu_id)
    print('use_gpu: ', args.use_gpu)
    if args.use_gpu:
        torch.cuda.set_device(device=args.gpu_id)
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
