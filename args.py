import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='MIMN')

    parser.add_argument('--imsize', dest='imsize', default=352, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=4, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=1,type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', default=50, type=int)
    parser.add_argument('--print_every', dest='print_every', default=10,type=int)

    # GPU
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('--gpu_id', dest='gpu_id', default=0, type=int)

    # model prepare
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help=('whether to resume training an existing model'
                              '(the one with name model_name will be used)'))
    parser.set_defaults(resume=False)
    parser.add_argument('--epoch_resume', dest='epoch_resume', default=0, type=int)
    parser.add_argument('--model_name', dest='model_name', default='MIMN_model')
    parser.add_argument('--rgb_memory_path', help='path to store rgb memory', default='ckpt/memory_rgb.pth')
    parser.add_argument('--flow_memory_path', help='path to store flow memory', default='ckpt/memory_flow.pth')
    
    # pretrained backbone model
    parser.add_argument('--checkpoint', dest='checkpoint', default='MIMN_backbone')
    parser.add_argument('--ckptEpoch', dest='ckptEpoch', default=39, type=int)

    # training parameters
    parser.add_argument('--root', help='dataset path', default='dataset/')
    parser.add_argument('--seed', dest='seed', default=123, type=int)
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--lr_cnn', dest='lr_cnn', default=1e-4, type=float)
    parser.add_argument('--optim_cnn', dest='optim_cnn', default='sgd',
                        choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-5,
                        type=float)
    parser.add_argument('--weight_decay_cnn', dest='weight_decay_cnn',
                        default=1e-5, type=float)
    parser.add_argument('--optim', dest='optim', default='sgd',
                        choices=['adam', 'sgd', 'rmsprop'])

    return parser


if __name__ =="__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
