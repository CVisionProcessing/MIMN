import torch
import os
import pickle
from collections import OrderedDict


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_parallel(encoder_dict, decoder_dict):
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict

def get_optimizer(optim_name, lr, parameters, weight_decay=0, momentum=0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                              lr=lr, weight_decay=weight_decay,
                              momentum=momentum)
    elif optim_name == 'adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                               lr=lr, weight_decay=weight_decay)
    return opt

def save_checkpoint_epoch(args, encoder, decoder, enc_opt, dec_opt, epoch, best=False):
    torch.save(encoder.state_dict(), os.path.join('ckpt', args.model_name, 'encoder_{}.pt'.format(epoch)))
    torch.save(decoder.state_dict(), os.path.join('ckpt', args.model_name, 'decoder_{}.pt'.format(epoch)))
    torch.save(enc_opt.state_dict(), os.path.join('ckpt', args.model_name, 'enc_opt_{}.pt'.format(epoch)))
    torch.save(dec_opt.state_dict(), os.path.join('ckpt', args.model_name, 'dec_opt_{}.pt'.format(epoch)))

    if best:
        torch.save(encoder.state_dict(), os.path.join('ckpt', args.model_name, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join('ckpt', args.model_name, 'decoder.pt'))
        torch.save(enc_opt.state_dict(), os.path.join('ckpt', args.model_name, 'enc_opt.pt'))
        torch.save(dec_opt.state_dict(), os.path.join('ckpt', args.model_name, 'dec_opt.pt'))

    # save parameters for future use
    pickle.dump(args, open(os.path.join('ckpt', args.model_name, 'args.pkl'), 'wb'))


def load_checkpoint_epoch(model_name, epoch, use_gpu=True, load_opt=True):
    if use_gpu:
        encoder_dict = torch.load(os.path.join('ckpt', model_name, 'encoder_{}.pt'.format(epoch)), map_location='cuda:0')
        decoder_dict = torch.load(os.path.join('ckpt', model_name, 'decoder_{}.pt'.format(epoch)), map_location='cuda:0')
        if load_opt:
            enc_opt_dict = torch.load(os.path.join('ckpt', model_name, 'enc_opt_{}.pt'.format(epoch)))
            dec_opt_dict = torch.load(os.path.join('ckpt', model_name, 'dec_opt_{}.pt'.format(epoch)))
    else:
        encoder_dict = torch.load(os.path.join('ckpt', model_name, 'encoder_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('ckpt', model_name, 'decoder_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join('ckpt', model_name, 'enc_opt_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join('ckpt', model_name, 'dec_opt_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
    # save parameters for future use
    if load_opt:
        args = pickle.load(open(os.path.join('ckpt', model_name, 'args.pkl'), 'rb'))

        return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args
    else:
        return encoder_dict, decoder_dict, None, None, None
