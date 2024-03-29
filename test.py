import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
import time
import torch
from torchvision import transforms
import argparse

import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

from imageio import imread
from MIMN import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch
import torch.nn.functional as F
from dataset_pro import VSKTestDataset

import cv2
from matplotlib import pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser(description="MIMN")
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--model_name", type=str, default="MIMN", help="model name")
    parser.add_argument("--epoch", type=str, default="89", help="model epoch")
    parser.add_argument('--root', help='dataset path', default='dataset/')
    parser.add_argument("--dataset", type=str, default='DAVIS', help="DAVIS, FBMS, SegTrackv2")
    parser.add_argument("--size", type=int, default=352, help="resize image to the model")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--result_path", type=str, default="./output")
    parser.add_argument("--rgb_memory_path", default="ckpt/rgb_memory_test.pth")
    parser.add_argument("--flow_memory_path", default="ckpt/flow_memory_test.pth")
    return parser.parse_args()

def testIters(args):
    model_name = args.model_name
    epoch = args.epoch
    result_dir = os.path.join(args.result_path, args.dataset)
    
    encoder_dict, decoder_dict, _, _, _ =\
        load_checkpoint_epoch(model_name, epoch, True, False)
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    encoder.cuda()
    decoder.cuda()

    # Testing Data
    test_path = os.path.join(args.root, args.dataset)
    dataset = VSKTestDataset(test_path, args.size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Testing Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset), len(dataloader)))
    
    
    start = time.time()
    with torch.no_grad():
        with open(args.rgb_memory_path,'w') as f:
            pass
        with open(args.flow_memory_path,'w') as f:
            pass
        for data in dataloader:
            rgb_last_memory = torch.load(args.rgb_memory_path) if os.path.getsize(args.rgb_memory_path)!=0 else {}
            flow_last_memory = torch.load(args.flow_memory_path) if os.path.getsize(args.flow_memory_path)!=0 else {}
            
            image, flow, _class, seq, name, inputSize = data['video'].cuda(), data['fwflow'].cuda(), data['_class'], data['seq'], data['name'], data['inputSize']
            B, C, H, W = image.size()
            x, spatial_feature, y, temporal_feature = encoder(image, flow)
            out, rgb_new_memory, flow_new_memory = decoder(x, spatial_feature, y, temporal_feature, rgb_last_memory, flow_last_memory, _class)
            torch.save(rgb_new_memory, args.rgb_memory_path)
            torch.save(flow_new_memory, args.flow_memory_path)
            
            for i in range(len(out)):
                mask_pred = out[i, 0, :, :]
                mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')
                save_folder = os.path.join(result_dir, seq[i])
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_file = os.path.join(save_folder, name[i] + '.png')
                mask_pred = mask_pred.resize((inputSize[0][i], inputSize[1][i]))
                mask_pred.save(save_file)

    end = time.time()
    total_time = end-start
    print('total_time: ' + str(total_time) + ', Fps: ' + str(len(dataset) / total_time))

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    if args.cuda:
        torch.cuda.set_device(device=args.gpus)
    testIters(args)