from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core.raft import RAFT
import evaluate
import datasets
# from FeatureProjection import*
from torch.utils.tensorboard import SummaryWriter
# import kd_utils 
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 1000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}


        self.writer=None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
   
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
            

        for key in results:
       
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
    
        self.writer.close()
def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def train(s_args,t_args):
    #define networks,pass data through each network and return attention maps for context layer
    #pass context attention maps through paraphraser and translator
    #add new loss 
    #adjust loss ratios
    # print('test')
    # paraphraser=Paraphraser(64, 64)
    # paraphraser.load_state_dict(torch.load('ckpt/Paraphraser/paraphraser/paraphraser_5.pth'), strict=False)
    # translator=Translator(64, 64)
    teacher = nn.DataParallel(RAFT(t_args), device_ids=t_args.gpus)
    student=nn.DataParallel(RAFT(s_args), device_ids=s_args.gpus)

    print("Parameter Count Teacher: %d" % count_parameters(teacher))
    print("Parameter Count Teacher: %d" % count_parameters(student))

    if t_args.restore_ckpt is not None:
        teacher.load_state_dict(torch.load(t_args.restore_ckpt), strict=False)
    if s_args.restore_ckpt is not None:
        student.load_state_dict(torch.load(s_args.restore_ckpt), strict=False)

    teacher.cuda()
    teacher.eval()

    student.cuda()
    student.train()

    # paraphraser.cuda()
    # paraphraser.eval()

    # translator.cuda()
    # translator.train()

    # if t_args.stage != 'chairs':
    #     teacher.module.freeze_bn()

    if s_args.stage != 'chairs':
      student.module.freeze_bn()


    train_loader = datasets.fetch_dataloader(s_args)
    print(len(train_loader))
    optimizer, scheduler = fetch_optimizer(s_args, student)
    # optimizer_module,scheduler_module=fetch_optimizer(s_args,translator)
    # optimizer_module = optim.SGD(translator.parameters(), lr=s_args.lr, momentum=0.9, weight_decay=s_args.wdecay)
    # scheduler_module = optim.lr_scheduler.MultiStepLR(optimizer_module, milestones=[10,20], gamma=0.1)


    criterion_l1 = nn.L1Loss()
    total_steps = 0
    scaler = GradScaler(enabled=s_args.mixed_precision)
    
    logger = Logger(student, scheduler)

    VAL_FREQ = 5000
    add_noise = True
    print(measure_module_sparsity(student))
    for module_name,module in student.named_modules():

      # print(module)
      if isinstance(module, torch.nn.Conv2d):

        prune.ln_structured(module, name="weight", amount=0.5,n=2, dim=0)
        prune.remove(module, 'weight')
        
    print(measure_module_sparsity(student))

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # optimizer_module.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if s_args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                

            # (flow_predictions,attention_map_1),(attention_map_2,attention_map_3)
            t_flow_predictions, t_context_attention_map= teacher(image1, image2, iters=t_args.iters)  

            s_flow_predictions, s_context_attention_map= student(image1, image2, iters=s_args.iters) 
            
            # print('attention map break')
            # factor_t = paraphraser(t_context_attention_map[0],1);
            # factor_s = translator(s_context_attention_map[0]);
            #attention_map is list of two convolutional layers 
            BETA=5000
            # beta_loss = BETA * (criterion_l1(kd_utils.FT(factor_s), kd_utils.FT(factor_t.detach()))) 
           
           

            loss, metrics = sequence_loss(s_flow_predictions, flow, valid, s_args.gamma)
            # print(loss.item(),beta_loss.item())
            beta_loss=0
            loss = loss+beta_loss
          
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  
            # scaler.unscale_(optimizer_module)              
            torch.nn.utils.clip_grad_norm_(student.parameters(), s_args.clip)
            
            scaler.step(optimizer)
            # scaler.step(optimizer_module)
            scheduler.step()
            # scheduler_module.step()
            scaler.update()
            # fix optimizer and scheduler for translator
            
            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
            
                PATH = 'checkpoints/pruned_sintel/%d_%s.pth' % (total_steps+1, s_args.name)
                torch.save(student.state_dict(), PATH)

                results = {}
                for val_dataset in s_args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(student.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(student.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(student.module))

                logger.write_dict(results)
                
                student.train()
                if s_args.stage != 'chairs':
                    student.module.freeze_bn()
            
            total_steps += 1

            if total_steps > s_args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/pruned_sintel/%s.pth' % s_args.name
    torch.save(student.state_dict(), PATH)

    return PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    t_args = parser.parse_args()
    
    s_args=parser.parse_args(['--stage','sintel','--validation','sintel',
                             '--num_steps','5000 ','--batch_size','2',
                             '--lr','0.00001 ','--image_size','368','768 ',
                             '--wdecay','0.00001 ','--gamma','0.85','--small',
                             '--name','raft_small_student_sintel', '--gpus','0',
                             '--restore_ckpt', 'checkpoints/sintel/100000_raft_small_student_sintel.pth',
                            ])
    print(s_args)
    print(t_args)
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(s_args,t_args)