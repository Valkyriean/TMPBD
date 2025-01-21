from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
from spikingjelly.activation_based import functional, neuron, monitor
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from models import get_model
from scipy.stats import gamma
import os
import csv
from torch.nn.functional import avg_pool1d, max_pool1d
import cupy




def get_data_meta(dataset, npara):
    if dataset == 'gesture':
        input_dim = (16, npara, 2, 128, 128)
        NC = 11
        penultimate_neuron = neuron.LIFNode

    elif dataset == 'cifar10':
        input_dim = (16, npara, 2, 128, 128)
        NC = 10
        penultimate_neuron = neuron.LIFNode

    elif dataset == 'mnist':
        input_dim = (16, npara, 2, 34, 34)
        NC = 10
        penultimate_neuron = neuron.IFNode
    elif dataset == 'caltech':
        input_dim = (16, npara, 2, 180, 180)
        NC = 101
        penultimate_neuron = neuron.LIFNode
    return input_dim, NC, penultimate_neuron


def lr_scheduler(iter_idx):
    lr = 1e-2
    return lr


def save_loss_plot(label, loss_values,res_values, model_name):
    save_dir = f'./plots/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(range(len(loss_values)), loss_values, label='Loss')
    plt.xlabel('Iteration Index')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Iteration for label: {label})')
    plt.legend()

    save_path = f'{save_dir}/loss_vs_iter_{label}.png'

    plt.savefig(save_path)
    plt.close() 
    print(f'Loss plot saved at: {save_path}')

    # RES plot
    plt.figure()
    plt.plot(range(len(res_values)), res_values, label='Res')
    plt.xlabel('Iteration Index')
    plt.ylabel('Res')
    plt.title(f'Res vs Iteration for label: {label})')
    plt.legend()

    save_path = f'{save_dir}/res_vs_iter_{label}.png'

    plt.savefig(save_path)
    plt.close()  
    print(f'Res plot saved at: {save_path}')


def backdoor_detection(model, img_dim, NC, nstep, penultimate_neuron, model_name, args, device):
    res = []
    for t in range(NC):
        # images = torch.randint(0, 11, input_dim).float().to(device)
        # images = torch.zeros(input_dim).to(device)
        
        images = torch.rand(img_dim).to(device)
        images.requires_grad = True
        last_loss = 1000
        labels = t * torch.ones((len(images[0]),), dtype=torch.long).to(device)
        onehot_label = F.one_hot(labels, num_classes=NC)
        loss_values = []
        res_values = []
        for iter_idx in tqdm(range(nstep),disable=True):
            optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
            optimizer.zero_grad()
            functional.reset_net(model)
            
            v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=model, instance=penultimate_neuron)

            outputs_fr = model(torch.clamp(images, min=0, max=10)).mean(0)
            
            
            v = v_seq_monitor.records[-1].mean(0)
            v_max = torch.max(v_seq_monitor.records[-1], dim=0)[0]
            v_seq_monitor.clear_recorded_data()
            v_seq_monitor.remove_hooks()
            del v_seq_monitor
            
  
            if args.target == "v_avg":
                outputs = avg_pool1d(v, kernel_size=10, stride=10)
                # print(outputs.shape)
            elif args.target == "fr":
                outputs = outputs_fr
            elif args.target == "v_max":
                outputs = max_pool1d(v_max, kernel_size=10, stride=10)

            loss = -1 * torch.sum((outputs * onehot_label)) \
                   + torch.sum(torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
                   
            
            loss.backward(retain_graph=True)
            optimizer.step()
            if args.save_plot:
                loss_values.append(loss.item())
                res_values.append(torch.max(torch.sum((outputs * onehot_label), dim=1) \
                             - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
            if args.early_stop and (last_loss == 0 or abs(last_loss - loss.item())/abs(last_loss)< 1e-5):
                print(f'End at run {iter_idx}, loss {loss.item()}, last loss {last_loss}')
                break
            last_loss = loss.item()
        if args.save_plot:
            save_loss_plot(t, loss_values, res_values, model_name)
        res.append(torch.max(torch.sum((outputs * onehot_label), dim=1) \
                             - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
        print(t, res[-1])

    stats = res

    ind_max = np.argmax(stats)
    r_eval = np.amax(stats)
    r_null = np.delete(stats, ind_max)
    
    shape, loc, scale = gamma.fit(r_null)
    pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null) + 1)
    print(pv)
    if pv > 0.05:
        print('No Attack!')
        return -1, pv
    else:
        print('There is attack with target class {}'.format(np.argmax(stats)))
        return np.argmax(stats), pv


def save_result(nstep, img_dim, dataset, attack_type, id, true_label, pred_label,elapsed_time, pv, correct):
    save_path = "experiments"
    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/detection_results.csv'.format(save_path)
    header = ['NSTEP', 'IMG DIM', 'Dataset', 'Attack Type', 'Model ID', 'True Label', 'Predicted Label', 'Elapsed Time', 'P-Value',
              'Correctness']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([nstep, img_dim, dataset, attack_type, id, true_label, pred_label,elapsed_time, pv, correct])



def load_detection(args, device):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    start_time = time.time()  # 开始计时
    input_dim, NC, penultimate_neuron = get_data_meta(args.dataset, args.npara)
    model = get_model(args.dataset, 16)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    for m in model.modules():
        if isinstance(m, penultimate_neuron):
            m.store_v_seq = True
    # if device == 'cuda':
    #    model = torch.nn.DataParallel(model)
    #    cudnn.benchmark = True
    if args.model_name:
        model_path = f'{args.model_dir}/{args.model_name}.pth'
    else:
        args.model_name = f'{args.dataset}-{args.attack_type}-{args.attack_label}'
        model_path = f'{args.model_dir}/{args.dataset}-{args.attack_type}-{args.attack_label}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))

    # model.eval()
    functional.set_step_mode(model, 'm')
    if torch.cuda.is_available():
        functional.set_backend(model, 'cupy', instance=penultimate_neuron)
        cupy.random.seed(np.uint64(args.seed))
    # img_dim = (input_dim[0], input_dim[3] , input_dim[2], input_dim[3], input_dim[4])
    # nstep = input_dim[0]*input_dim[2]*input_dim[3]
    img_dim = input_dim

    predict_label, pv = backdoor_detection(model, img_dim, NC, args.nstep, penultimate_neuron, args.model_name, args, device)

    if args.attack_type == 'clean':
        true_label = -1
    else:
        true_label = args.attack_label

    correct = (predict_label == true_label)
    end_time = time.time()  # Stop timing

    elapsed_time = end_time - start_time
    print(f"Program running for: {elapsed_time:.6f} seconds")

    save_result(args.nstep,img_dim, args.dataset, args.attack_type, args.attack_label, true_label, predict_label,elapsed_time, pv, correct)
    return correct
    
    
device = 'cuda' if torch.cuda.is_available() else 'mps'


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='SNN MMBD')
    parser.add_argument('--model_dir',type=str, default='models', help='model path')
    parser.add_argument('--dataset', '-d', type=str, default='gesture', choices=["gesture", "cifar10", "caltech", "mnist"], help='dataset name')
    parser.add_argument('--attack_type', '-t', type=str, default='clean', choices=["clean", "static", "dynamic", "moving"], help='attack type')
    parser.add_argument('--attack_label', '-l',type=int, default=0, help='attack label')
    parser.add_argument('--model_name', type=str, default=None, help='model name')


    parser.add_argument('--nstep', type=int, default=5000, help='max optimisation epoches')
    parser.add_argument('--npara', type=int, default=3, help='number of parallel initial states')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--early_stop', action='store_true',help='to use early stop')
    parser.add_argument('--save_plot', action='store_true',help='save the learning graph')

    parser.add_argument('--target', type=str, default="v_avg",choices=["fr", "v_avg", "v_max", "h"] ,help='which variable targeted to be optimised')

    args = parser.parse_args()
    load_detection(args, device)
