import copy
import torch
import torch.nn as nn
from models import Autoencoder, get_model
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import cupy
import torch.optim.lr_scheduler as lr_scheduler
import random, os, csv
from tqdm import tqdm
import numpy as np
from datasets import get_dataset
from poisoned_dataset import PoisonedDataset
from utils import get_data_meta, split_dataset_by_class, evaluate, dynamic_evaluate, remove_attack_label_samples, clip_image
from detect import backdoor_detection


parser = argparse.ArgumentParser(description='SNN MMBD')
parser.add_argument('--model_dir',type=str, default='models', help='model path')
parser.add_argument('--dataset', '-d', type=str, default='gesture', choices=["gesture", "cifar10", "caltech", "mnist"], help='dataset name')
parser.add_argument('--attack_type', '-t', type=str, default='static', choices=["clean", "static", "dynamic", "moving", "smart"], help='attack type')
parser.add_argument('--attack_label', '-l',type=int, default=0, help='attack label')
parser.add_argument('--model_name', type=str, default=None, help='model name')


parser.add_argument('--nstep', type=int, default=50, help='max optimisation epoches')
parser.add_argument('--nsample', type=int, default=20, help='Number of samples per class')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--early_stop', action='store_true',help='to use early stop')
parser.add_argument('--save_plot', action='store_true',help='save the learning graph')

parser.add_argument('--clamp_method', type=str, default="clamp",choices=["clamp", "abs", "max"] ,help='The method to clamp the input')
parser.add_argument('--algorithm', type=str, default="clamping",choices=["clamping", "finetuning", "self_tuning", "original", "mmbm", "e2e"] ,help='The algorithm to mitigate the attack')
parser.add_argument('--target', type=str, default="v_avg",choices=["fr", "v_avg", "v_max", "h"] ,help='which variable targeted to be optimised')


args = parser.parse_args()


class Args():
    dataset = args.dataset
    lr = 0.001
    batch_size = 1
    epochs = 10
    seed = 42
    T = 16
    amp = False
    cupy = True
    loss = 'mse'
    optim = 'adam'
    trigger_label = args.attack_label
    polarity = 1
    trigger_size = 0.1
    epsilon = 0.1
    pos='top-left'
    type = args.attack_type
    n_masks=2
    least = False
    most_polarity = False
    momentum = 0.9
    data_dir = 'data'
    save_path = 'experiment'
    model_path = ''
    frame_gap = 1


poison_args = Args()




class CleanDVSGestureNet(nn.Module):
    def __init__(self, device, model, init_dis=10, clamping_method="clamp"):
        super(CleanDVSGestureNet, self).__init__()
        self.clamping_method = clamping_method
        functional.reset_net(model)
        # model.eval()
        self.model = model
        
        self.clamp_w1 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis  
        self.clamp_w2 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis 
        self.clamp_w3 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis  
        self.clamp_w4 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis 
        self.clamp_w5 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis  
        
        
        self.clamp_m1 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis  
        self.clamp_m2 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis 
        self.clamp_m3 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis  
        self.clamp_m4 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis  
        self.clamp_m5 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis  
        
        self.clamp_w1.requires_grad = True
        self.clamp_w2.requires_grad = True
        self.clamp_w3.requires_grad = True
        self.clamp_w4.requires_grad = True
        self.clamp_w5.requires_grad = True
        self.clamp_m1.requires_grad = True
        self.clamp_m2.requires_grad = True
        self.clamp_m3.requires_grad = True
        self.clamp_m4.requires_grad = True
        self.clamp_m5.requires_grad = True

    def forward(self, x):
        out = self.model.conv_fc[1](self.model.conv_fc[0](x))  # Conv2d -> BatchNorm2d
        # print(out.shape)
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w1, max=self.clamp_w1)  # Absolute value clamping
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m1, max=self.clamp_w1)
        else:
            out = torch.min(out, self.clamp_w1) # Clamping
        # print(out.shape)
        out = self.model.conv_fc[2](out)  # LIFNode
        out = self.model.conv_fc[3](out)  # MaxPool2d

        out = self.model.conv_fc[5](self.model.conv_fc[4](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w2, max=self.clamp_w2)  # Absolute value clamping
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m2, max=self.clamp_w2)
        else:
            out = torch.min(out, self.clamp_w2) # Clamping        out = self.model.conv_fc[6](out)  # LIFNode
        out = self.model.conv_fc[7](out)  # MaxPool2d

        out = self.model.conv_fc[9](self.model.conv_fc[8](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w3, max=self.clamp_w3)  # Absolute value clamping
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m3, max=self.clamp_w3)
        else:
            out = torch.min(out, self.clamp_w3) # Clamping
        out = self.model.conv_fc[10](out)  # LIFNode
        out = self.model.conv_fc[11](out)  # MaxPool2d

        out = self.model.conv_fc[13](self.model.conv_fc[12](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w4, max=self.clamp_w4)  # Absolute value clamping
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m4, max=self.clamp_w4)
        else:
            out = torch.min(out, self.clamp_w4) # Clamping
        out = self.model.conv_fc[14](out)  # LIFNode
        out = self.model.conv_fc[15](out)  # MaxPool2d

        out = self.model.conv_fc[17](self.model.conv_fc[16](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w5, max=self.clamp_w5)  # Absolute value clamping
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m5, max=self.clamp_w5)
        else:
            out = torch.min(out, self.clamp_w5) # Clamping
        out = self.model.conv_fc[18](out)  # LIFNode
        out = self.model.conv_fc[19](out)  # MaxPool2d

        out = self.model.conv_fc[20](out)  # Flatten
        out = self.model.conv_fc[21](out)  # Dropout
        out = self.model.conv_fc[22](out)  # Linear (in_features=2048, out_features=512)
        out = self.model.conv_fc[23](out)  # LIFNode
        out = self.model.conv_fc[24](out)  # Dropout
        out = self.model.conv_fc[25](out)  # Linear (in_features=512, out_features=110)
        out = self.model.conv_fc[26](out)  # LIFNode
        out = self.model.conv_fc[27](out)  # VotingLayer

        return out

class CleanCIFAR10DVSNet(nn.Module):
    def __init__(self, device, model, init_dis=10, clamping_method="clamp"):
        """
        CleanCIFAR10DVSNet is modeled after CleanDVSGestureNet, applying a clamping-based 
        backdoor mitigation strategy to a CIFAR10DVSNet-like architecture.

        Args:
            device: The torch device (e.g. 'cuda' or 'cpu').
            model: A CIFAR10DVSNet model instance whose layers we will wrap with clamping.
            init_dis (float): Initial displacement for the clamp range (default 10).
            clamping_method (str): 'abs', 'clamp', or 'min' clamping approach.
        """
        super(CleanCIFAR10DVSNet, self).__init__()
        self.clamping_method = clamping_method
        functional.reset_net(model)
        self.model = model

        # ----------------------------------------
        # Define clamp parameters for 4 conv blocks
        # ----------------------------------------
        # The shape used in your CleanDVSGestureNet was [16, poison_args.batch_size, 128, 1, 1].
        # Here, we keep the same shape pattern but only define 4 blocks: w1..w4, m1..m4
        # If your channel or batch sizes differ, adjust these accordingly.
        self.clamp_w1 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis
        self.clamp_w2 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis
        self.clamp_w3 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis
        self.clamp_w4 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) + init_dis

        self.clamp_m1 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis
        self.clamp_m2 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis
        self.clamp_m3 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis
        self.clamp_m4 = torch.ones([16, poison_args.batch_size, 128, 1, 1]).to(device) - init_dis

        # ----------------------------------------
        # Set the clamp parameters to be learnable
        # ----------------------------------------
        for tensor in [
            self.clamp_w1, self.clamp_w2, self.clamp_w3, self.clamp_w4,
            self.clamp_m1, self.clamp_m2, self.clamp_m3, self.clamp_m4
        ]:
            tensor.requires_grad = True

    def forward(self, x):
        """
        Forward pass with clamping after each Conv+BN step (before LIF and Pool).
        Reproduces the CIFAR10DVSNet structure of 4 conv blocks, then the FC block.
        """
        # ---- Block 1: conv -> BN -> clamp -> LIF -> MaxPool
        out = self.model.conv_fc[1](self.model.conv_fc[0](x))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w1, max=self.clamp_w1)
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m1, max=self.clamp_w1)
        else:
            out = torch.min(out, self.clamp_w1)  # simplistic clamp
        out = self.model.conv_fc[2](out)  # LIF
        out = self.model.conv_fc[3](out)  # MaxPool2d

        # ---- Block 2: conv -> BN -> clamp -> LIF -> MaxPool
        out = self.model.conv_fc[5](self.model.conv_fc[4](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w2, max=self.clamp_w2)
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m2, max=self.clamp_w2)
        else:
            out = torch.min(out, self.clamp_w2)
        out = self.model.conv_fc[6](out)  # LIF
        out = self.model.conv_fc[7](out)  # MaxPool2d

        # ---- Block 3: conv -> BN -> clamp -> LIF -> MaxPool
        out = self.model.conv_fc[9](self.model.conv_fc[8](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w3, max=self.clamp_w3)
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m3, max=self.clamp_w3)
        else:
            out = torch.min(out, self.clamp_w3)
        out = self.model.conv_fc[10](out)  # LIF
        out = self.model.conv_fc[11](out)  # MaxPool2d

        # ---- Block 4: conv -> BN -> clamp -> LIF -> MaxPool
        out = self.model.conv_fc[13](self.model.conv_fc[12](out))  # Conv2d -> BatchNorm2d
        if self.clamping_method == "abs":
            out = torch.clamp(out, min=-self.clamp_w4, max=self.clamp_w4)
        elif self.clamping_method == "clamp":
            out = torch.clamp(out, min=self.clamp_m4, max=self.clamp_w4)
        else:
            out = torch.min(out, self.clamp_w4)
        out = self.model.conv_fc[14](out)  # LIF
        out = self.model.conv_fc[15](out)  # MaxPool2d

        # ---- Fully Connected block
        out = self.model.conv_fc[16](out)  # Flatten
        out = self.model.conv_fc[17](out)  # Dropout
        out = self.model.conv_fc[18](out)  # Linear (in_features=channels * 8 * 8, out_features=512)
        out = self.model.conv_fc[19](out)  # LIF
        out = self.model.conv_fc[20](out)  # Dropout
        out = self.model.conv_fc[21](out)  # Linear (in_features=512, out_features=100)
        out = self.model.conv_fc[22](out)  # LIF
        out = self.model.conv_fc[23](out)  # VotingLayer (10)

        return out

def evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, attack_type, attack_label, device):
    criterion = nn.MSELoss()
    if attack_type == "dynamic":
        _, ca = evaluate(model, clean_test_loader, criterion, device)
        _, asr = dynamic_evaluate(model, atkmodel, poisoned_test_loader,attack_label, device)
    else:
        _, ca = evaluate(model, clean_test_loader, criterion, device)
        _, asr = evaluate(model, poisoned_test_loader, criterion, device)
    return ca, asr


def clamping(dataset, model,atkmodel, train_loader,clean_test_loader,poisoned_test_loader , attack_label, device, epochs=50, lr=0.1, c=1e-5, min_c=1e-6, a=1.0, baseline_accuracy=85):
    if (dataset == "gesture"):
        clean_model = CleanDVSGestureNet(device, model, init_dis=10, clamping_method=args.clamp_method)
        optimizer = torch.optim.Adam([clean_model.clamp_w1, clean_model.clamp_w2, clean_model.clamp_w3, clean_model.clamp_w4, clean_model.clamp_w5, clean_model.clamp_m1, clean_model.clamp_m2, clean_model.clamp_m3, clean_model.clamp_m4, clean_model.clamp_m5], lr=lr)

    elif (dataset == "cifar10"):
        clean_model = CleanCIFAR10DVSNet(device, model, init_dis=20, clamping_method=args.clamp_method)
        optimizer = torch.optim.Adam([clean_model.clamp_w1, clean_model.clamp_w2, clean_model.clamp_w3, clean_model.clamp_w4, clean_model.clamp_m1, clean_model.clamp_m2, clean_model.clamp_m3, clean_model.clamp_m4], lr=lr)
        c = 2*c
    clean_model.to(device)
    for m in clean_model.modules():
        if isinstance(m, neuron.LIFNode):
            m.store_v_seq = True
    clean_model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                ref_v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=clean_model.model, instance=neuron.LIFNode)
                ref_out = clean_model.model(inputs.transpose(0, 1)).mean(0)
                ref_v = ref_v_seq_monitor.records[-1].mean(0)
                ref_v_seq_monitor.clear_recorded_data()
                ref_v_seq_monitor.remove_hooks()
                del ref_v_seq_monitor
            
            v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=clean_model, instance=neuron.LIFNode)
            outputs = clean_model(inputs.transpose(0, 1)).mean(0)
            v = v_seq_monitor.records[-1].mean(0)
            v_seq_monitor.clear_recorded_data()
            v_seq_monitor.remove_hooks()
            del v_seq_monitor

            loss1 = criterion(v, ref_v)
            if dataset == "gesture":
                loss2 = torch.norm(clean_model.clamp_w1) + torch.norm(clean_model.clamp_w2) + torch.norm(clean_model.clamp_w3) + torch.norm(clean_model.clamp_w4) + torch.norm(clean_model.clamp_w5) + torch.norm(clean_model.clamp_m1) + torch.norm(clean_model.clamp_m2) + torch.norm(clean_model.clamp_m3) + torch.norm(clean_model.clamp_m4) + torch.norm(clean_model.clamp_m5)
            elif dataset == "cifar10":
                loss2 = torch.norm(clean_model.clamp_w1) + torch.norm(clean_model.clamp_w2) + torch.norm(clean_model.clamp_w3) + torch.norm(clean_model.clamp_w4) + torch.norm(clean_model.clamp_m1) + torch.norm(clean_model.clamp_m2) + torch.norm(clean_model.clamp_m3) + torch.norm(clean_model.clamp_m4)
            loss = loss1 + c * loss2

            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.detach().item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels.argmax(1)).sum().item()
            total += labels.size(0)
            functional.reset_net(clean_model)
            functional.reset_net(clean_model.model)

        train_acc = 100. * correct / total
        # print(f"Epoch {epoch+1}/{epochs}, Loss 1: {loss1.item():.4f}, Loss 2: {c *loss2.item():.4f},Loss T: {loss.detach().item():.4f},Training Accuracy: {train_acc:.2f}%")
        



        ca, asr = evaluate_ca_asr(clean_model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
        print(f'CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')

        # print(f'Train test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%')

    
    ca, asr = evaluate_ca_asr(clean_model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
    return ca, asr, clean_model



def mmbm(dataset, model,atkmodel, train_loader,clean_test_loader,poisoned_test_loader , attack_label, device, epochs=50, lr=0.1, c=1e-5, min_c=1e-6, a=1.2, baseline_accuracy=85):
    if (dataset == "gesture"):
        clean_model = CleanDVSGestureNet(device, model, init_dis=10, clamping_method="max")
        optimizer = torch.optim.Adam([clean_model.clamp_w1, clean_model.clamp_w2, clean_model.clamp_w3, clean_model.clamp_w4, clean_model.clamp_w5, clean_model.clamp_m1, clean_model.clamp_m2, clean_model.clamp_m3, clean_model.clamp_m4, clean_model.clamp_m5], lr=lr)

    elif (dataset == "cifar10"):
        clean_model = CleanCIFAR10DVSNet(device, model, init_dis=20, clamping_method="max")
        optimizer = torch.optim.Adam([clean_model.clamp_w1, clean_model.clamp_w2, clean_model.clamp_w3, clean_model.clamp_w4, clean_model.clamp_m1, clean_model.clamp_m2, clean_model.clamp_m3, clean_model.clamp_m4], lr=lr)
        c = 2*c
    clean_model.to(device)
    for m in clean_model.modules():
        if isinstance(m, neuron.LIFNode):
            m.store_v_seq = True
    clean_model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                ref_v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=clean_model.model, instance=neuron.LIFNode)
                ref_out = clean_model.model(inputs.transpose(0, 1)).mean(0)
                ref_v = ref_v_seq_monitor.records[-1].mean(0)
                ref_v_seq_monitor.clear_recorded_data()
                ref_v_seq_monitor.remove_hooks()
                del ref_v_seq_monitor
            
            v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=clean_model, instance=neuron.LIFNode)
            outputs = clean_model(inputs.transpose(0, 1)).mean(0)
            v = v_seq_monitor.records[-1].mean(0)
            v_seq_monitor.clear_recorded_data()
            v_seq_monitor.remove_hooks()
            del v_seq_monitor

            
            loss1 = criterion(outputs, ref_out)
            if dataset == "gesture":
                loss2 = torch.norm(clean_model.clamp_w1) + torch.norm(clean_model.clamp_w2) + torch.norm(clean_model.clamp_w3) + torch.norm(clean_model.clamp_w4) + torch.norm(clean_model.clamp_w5) + torch.norm(clean_model.clamp_m1) + torch.norm(clean_model.clamp_m2) + torch.norm(clean_model.clamp_m3) + torch.norm(clean_model.clamp_m4) + torch.norm(clean_model.clamp_m5)
            elif dataset == "cifar10":
                loss2 = torch.norm(clean_model.clamp_w1) + torch.norm(clean_model.clamp_w2) + torch.norm(clean_model.clamp_w3) + torch.norm(clean_model.clamp_w4) + torch.norm(clean_model.clamp_m1) + torch.norm(clean_model.clamp_m2) + torch.norm(clean_model.clamp_m3) + torch.norm(clean_model.clamp_m4)
            loss = loss1 + c * loss2

            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.detach().item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels.argmax(1)).sum().item()
            total += labels.size(0)
            functional.reset_net(clean_model)
            functional.reset_net(clean_model.model)

        train_acc = 100. * correct / total
        # print(f"Epoch {epoch+1}/{epochs}, Loss 1: {loss1.item():.4f}, Loss 2: {c *loss2.item():.4f},Loss T: {loss.detach().item():.4f},Training Accuracy: {train_acc:.2f}%")
        


        if epoch % 10 == 0:
            if train_acc >= baseline_accuracy * 0.95:
                c *= a  
            else:
                c /= a  

            # print(f"Adjusted c: {c:.6f}")

        # ca, asr = evaluate_ca_asr(clean_model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
        # print(f'CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')

        # print(f'Train test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%')

    
    ca, asr = evaluate_ca_asr(clean_model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
    return ca, asr

def finetuning(model,atkmodel, train_loader,clean_test_loader,poisoned_test_loader , attack_label, device, epochs=50):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    try:
        n_classes = len(clean_test_loader.dataset.classes)
    except:
        n_classes = 10
    for epoch in range(epochs):
        
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tqdm(train_loader):
            optimizer.zero_grad()
            frame = frame.to(device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)

            
            
            # print(label)
            if len(label.shape) == 1:
                label = F.one_hot(label, n_classes).float()

            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

            label = label.argmax(1)
            train_samples += label.numel()
            train_loss += loss.detach().item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

        train_loss /= train_samples
        train_acc /= train_samples

        # print(f"Epoch {epoch+1}/{epochs}, Loss T: {train_loss:.4f},Training Accuracy: {train_acc:.2f}")

    ca, asr = evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
    return ca, asr



def self_tuning(model,
                atkmodel,
                train_loader,
                clean_test_loader,
                poisoned_test_loader,
                args,
                device,
                epochs=50):
    """
    Fine-tune the model by using its own outputs as pseudo-labels.

    Args:
        model: The model to be fine-tuned.
        atkmodel: Attack model (used for evaluation).
        train_loader: DataLoader for training data (labels are ignored).
        clean_test_loader: DataLoader for clean testing data.
        poisoned_test_loader: DataLoader for poisoned testing data.
        args: Arguments or configuration object, 
              containing e.g. args.attack_type, args.attack_label, etc.
        device: The device (CPU/GPU) for computation.
        epochs (int): Number of fine-tuning epochs.

    Returns:
        ca, asr: clean accuracy and attack success rate after fine-tuning.
    """

    # Example: You can adjust the learning rate, optimizer, and loss function as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()

    # Just in case you want a default for number of classes
    # (not strictly needed if you're not using one-hot anymore)
    try:
        n_classes = len(clean_test_loader.dataset.classes)
    except:
        n_classes = 10

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        for frame, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to device
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)

            # ------------------------
            # 1) Get the model's output as pseudo-label (no_grad to avoid gradient flow)
            # ------------------------
            with torch.no_grad():
                out_fr_pseudo = model(frame).mean(0)  # shape: [N, n_classes]
            # Convert logits to a "soft" one-hot distribution
            pseudo_label = F.softmax(out_fr_pseudo, dim=1)

            # ------------------------
            # 2) Forward pass again to compute the loss against the pseudo-label
            # ------------------------
            optimizer.zero_grad()
            out_fr = model(frame).mean(0)  # shape: [N, n_classes]
            
            # Use MSELoss or CrossEntropy with soft labels (KLDivLoss), etc.
            loss = criterion(out_fr, pseudo_label)
            loss.backward()
            optimizer.step()

            # ------------------------
            # 3) Compute training metrics
            # ------------------------
            # For accuracy, we compare the argmax of the output with the argmax of pseudo-label
            label_argmax = pseudo_label.argmax(dim=1)
            train_samples += label_argmax.numel()
            train_loss += loss.detach().item() * label_argmax.numel()
            train_acc += (out_fr.argmax(dim=1) == label_argmax).float().sum().item()

            # Reset the spiking state if using spiking-based model
            functional.reset_net(model)

        # Average training loss & accuracy
        train_loss /= train_samples
        train_acc /= train_samples
        # print(f"[Epoch {epoch+1}/{epochs}] Loss: {train_loss:.4f} | Accuracy vs. pseudo-label: {train_acc:.4f}")

    # After fine-tuning, evaluate performance
    ca, asr = evaluate_ca_asr(model,
                              atkmodel,
                              clean_test_loader,
                              poisoned_test_loader,
                              args.attack_type,
                              args.attack_label,
                              device)
    return ca, asr

def save_result(dataset, attack_type, attack_label, algorithm, ca, asr):
    save_path = "experiments"
    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/mitigation.csv'.format(save_path)
    header = ['Dataset', 'Attack Type', 'Model ID', 'Algorithm', 'Clean Accuracy', 'Attack Success Rate']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, attack_type, attack_label, algorithm, f"{ca * 100:.2f}", f"{asr * 100:.2f}"])
        


def e2e_evaluate(model, clean_model, test_loader, predict_label, criterion, device):
    """
    1. First-pass inference with 'model'.
    2. If a sample's predicted label == 'predict_label', re-run that sample with 'clean_model'
       and overwrite the final prediction.
    3. Compute final accuracy on the entire test set.
    """

    model.eval()
    clean_model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0

    with torch.no_grad():
        for frame, label in tqdm(test_loader, disable=True):
            # Move to device and transpose for spiking jelly if needed
            frame = frame.to(device).transpose(0, 1)  # [N,T,C,H,W]->[T,N,C,H,W]
            label = label.to(device)

            # First inference: use the original model
            out_fr = model(frame).mean(0)        # shape: [N, #classes]
            loss = criterion(out_fr, label)      # label is assumed to be one-hot or something appropriate

            # Record stats
            test_loss += loss.detach().item() * label.size(0)
            test_samples += label.size(0)

            # Convert predictions to int labels
            pred = out_fr.argmax(dim=1)          # shape: [N]
            true_label = label.argmax(dim=1)     # shape: [N]

            # Second inference (clean model) for samples predicted == predict_label
            # 1) run entire batch on clean_model
            out_fr_clean = clean_model(frame).mean(0)
            pred_clean = out_fr_clean.argmax(dim=1)

            # print(f"pred{pred}, pred_clean{pred_clean}, true_label{true_label}")
            
            # 2) for each sample i, if pred[i] == predict_label, override with pred_clean[i]
            for i in range(len(pred)):
                if pred[i].item() == predict_label:
                    pred[i] = pred_clean[i]


            # print(f"pred{pred}, pred_clean{pred_clean}, true_label{true_label}, ")

            
            # Now compute accuracy
            test_acc += (pred == true_label).sum().item()

            # Reset states for spiking models
            functional.reset_net(model)
            functional.reset_net(clean_model)

    test_loss /= test_samples
    test_acc /= test_samples
    return test_loss, test_acc

def e2e_dynamic_evaluate(model, clean_model, atkmodel, test_loader, predict_label, attack_label, device):
    """
    Similar two-step inference approach for dynamic scenarios:
      - Normal inference pass (clean data).
      - Attack inference pass (triggered data).
    We measure:
      - test_acc = accuracy on clean data
      - test_asr = fraction of samples that end up predicted as 'attack_label'
                   when triggered with atkmodel.
    """

    model.eval()
    clean_model.eval()
    atkmodel.eval()

    # For convenience, get number of classes if needed
    try:
        n_classes = len(test_loader.dataset.classes)
    except:
        n_classes = 10

    total_samples = 0
    correct_clean = 0  # For normal accuracy
    success_attack = 0 # For ASR (how many get predicted as 'attack_label')

    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(device).transpose(0, 1)
            label = label.to(device)
            batch_size = label.size(0)
            total_samples += batch_size

            # ===== Clean (normal) inference =====
            out_fr = model(frame).mean(0)
            pred = out_fr.argmax(dim=1)
            true_label = label.argmax(dim=1)

            # Bulk second inference with clean_model
            out_fr_clean = clean_model(frame).mean(0)
            pred_clean = out_fr_clean.argmax(dim=1)

            # Overwrite predictions if they match predict_label
            for i in range(batch_size):
                if pred[i].item() == predict_label:
                    pred[i] = pred_clean[i]

            correct_clean += (pred == true_label).sum().item()

            # Reset states
            functional.reset_net(model)
            functional.reset_net(clean_model)

            # ===== Attack (trigger) inference =====
            # Use atkmodel to generate triggered input
            noise = atkmodel(frame)
            atkdata = clip_image(frame, noise, 0.01)  # or however you generate triggered data

            # First pass with original model
            out_fr_bk = model(atkdata).mean(0)
            pred_bk = out_fr_bk.argmax(dim=1)

            # Second pass with clean_model
            out_fr_bk_clean = clean_model(atkdata).mean(0)
            pred_bk_clean = out_fr_bk_clean.argmax(dim=1)

            for i in range(batch_size):
                if pred_bk[i].item() == predict_label:
                    pred_bk[i] = pred_bk_clean[i]

            # Count how many are predicted as attack_label
            success_attack += (pred_bk == attack_label).sum().item()

            # Reset states
            functional.reset_net(model)
            functional.reset_net(clean_model)
            functional.reset_net(atkmodel)


    test_acc = correct_clean / total_samples
    test_asr = success_attack / total_samples  # fraction predicted as 'attack_label'
    return test_acc, test_asr


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron

def e2e(dataset, model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, 
        attack_label, device, args):
    # 1) Try to load clamped model if it exists
    # try:
    #     clean_model = CleanDVSGestureNet(device, copy.deepcopy(model), init_dis=10, clamping_method=args.clamp_method)
    #     # Note: weights_only=True is available in recent PyTorch, else remove the param
    #     path = f"models/{args.dataset}-{args.attack_type}-{args.attack_label}-clamp.pth"
    #     print(path)
    #     clean_model.load_state_dict(
    #         torch.load(
    #             path,
    #             map_location=device
    #         ), 
    #         strict=True
    #     )
    #     clean_model.to(device)
    #     for m in clean_model.modules():
    #         if isinstance(m, neuron.LIFNode):
    #             m.store_v_seq = True
    #     print("Loaded clamped model from file.")
    # except:
        # If it fails, do the clamping procedure now
    print("Clamped model not found, clamping now...")
    ca, asr, clean_model = clamping(
        dataset, copy.deepcopy(model), atkmodel,
        train_loader, clean_test_loader, poisoned_test_loader,
        poison_args.trigger_label, device,
        epochs=args.nstep, lr=0.1, c=1e-5, min_c=1e-6,
        a=1.0, baseline_accuracy=85
    )
    torch.save(clean_model.state_dict(), f"models/{args.dataset}-{args.attack_type}-{args.attack_label}-clamp.pth")

    # 2) Attempt to read detection.csv and find the predicted label for (Dataset, Attack Type, Model ID)
    #    We'll assume you have something like args.l for Model ID
    detection_csv = "reporting/detection.csv"
    predict_label = None
    pv = None  # if you'd like, we can also read P-Value from CSV

    if os.path.exists(detection_csv):
        try:
            df = pd.read_csv(detection_csv)
            # We'll assume your code has an integer Model ID in args.l or something similar
            # so we can match the row by (Dataset, Attack Type, Model ID)
            # If your variable for model ID is different, adapt accordingly.
            cond = (
                (df["Dataset"] == args.dataset) &
                (df["Attack Type"] == args.attack_type) &
                (df["Model ID"] == args.attack_label)
            )
            matched_rows = df[cond]
            if len(matched_rows) > 0:
                # We'll just take the first match
                row = matched_rows.iloc[0]
                # "Predicted Label" is the column in your CSV
                csv_pred_label = row["Predicted Label"]
                # If the CSV is -1 or missing, we might treat that as no valid detection
                # if csv_pred_label != -1:
                predict_label = int(csv_pred_label)
                print(f"Found predicted label={predict_label} in detection.csv for Model ID={args.attack_label}")
                # If you also want P-Value or other info
                pv = row["P-Value"]
        except Exception as e:
            print(f"Warning: Could not parse detection.csv properly: {e}")

    # 3) If we didn't find a valid predicted label from CSV, fallback to backdoor_detection
    if predict_label is None:
        for m in model.modules():
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = True
        predict_label, pv = backdoor_detection(
            model, (16, 3, 2, 128, 128), 11, 5000, neuron.LIFNode, "", args, device
        )
        print(f"Detection fallback -> Predicted label: {predict_label}, P-Value: {pv}")
    else:
        print(f"Use detection.csv -> Predicted label: {predict_label}, P-Value: {pv}")

    criterion = nn.MSELoss()

        
    # ca, asr = evaluate_ca_asr(clean_model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
    # print(f'CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')
    # 4) Evaluate
    if args.attack_type == "dynamic":
        _, ca = e2e_evaluate(model, clean_model, clean_test_loader, predict_label, criterion, device)
        _, asr = e2e_dynamic_evaluate(model, clean_model, atkmodel, poisoned_test_loader, predict_label, attack_label, device)
    else:
        _, ca = e2e_evaluate(model, clean_model, clean_test_loader, predict_label, criterion, device)
        _, asr = e2e_evaluate(model, clean_model, poisoned_test_loader, predict_label, criterion, device)

    return ca, asr



def main():
    print(f"Algo{args.algorithm}, CM {args.clamp_method}, Type {args.attack_type}, Label {args.attack_label}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args.dataset, poison_args.T)
    if args.model_name:
        model_path = f'{args.model_dir}/{args.model_name}.pth'
    else:
        args.model_name = f'{args.dataset}-{args.attack_type}-{args.attack_label}'
        model_path = f'{args.model_dir}/{args.dataset}-{args.attack_type}-{args.attack_label}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
    model.to(device)

    model.eval()

    functional.set_step_mode(model, 'm')
    if torch.cuda.is_available():
        functional.set_backend(model, 'cupy', instance=neuron.LIFNode)
        cupy.random.seed(np.uint64(42))

    functional.reset_net(model)

     # Get the dataset
    _, test_data = get_dataset(poison_args.dataset, poison_args.T, poison_args.data_dir)




    test_data_clean = PoisonedDataset(test_data, poison_args.trigger_label, mode='test', epsilon=0,
                                    pos=poison_args.pos, attack_type=poison_args.type, time_step=poison_args.T,
                                    trigger_size=poison_args.trigger_size, dataname=poison_args.dataset,
                                    polarity=poison_args.polarity, n_masks=poison_args.n_masks, least=poison_args.least, most_polarity=poison_args.most_polarity, frame_gap = poison_args.frame_gap)



    train_set_clean, test_set_clean = split_dataset_by_class(test_data_clean, test_data_clean, args.nsample)
        


    # data in [16, 1, 2, 128, 128], where 16 is the time steps, 1 is the batch size, 2 is the channel, 128 is the height and 128 is the width
    # gesture data has 11 classes
    


    train_loader = DataLoader(dataset=train_set_clean, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)
    clean_test_loader = DataLoader(dataset=test_set_clean, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)
    # train_loader = DataLoader(dataset=train_set_clean, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)
    # clean_test_loader = DataLoader(dataset=test_data_clean, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)


    
    if args.attack_type != "dynamic":
        if args.attack_type == "clean":
            test_data_poisoned = PoisonedDataset(test_data, poison_args.trigger_label, mode='test', epsilon=1,
                                            pos=poison_args.pos, attack_type="static", time_step=poison_args.T,
                                            trigger_size=poison_args.trigger_size, dataname=poison_args.dataset,
                                            polarity=poison_args.polarity, n_masks=poison_args.n_masks, least=poison_args.least, most_polarity=poison_args.most_polarity, frame_gap = poison_args.frame_gap)

        else:
            test_data_poisoned = PoisonedDataset(test_data, poison_args.trigger_label, mode='test', epsilon=1,
                                            pos=poison_args.pos, attack_type=poison_args.type, time_step=poison_args.T,
                                            trigger_size=poison_args.trigger_size, dataname=poison_args.dataset,
                                            polarity=poison_args.polarity, n_masks=poison_args.n_masks, least=poison_args.least, most_polarity=poison_args.most_polarity, frame_gap = poison_args.frame_gap)

        train_set_poisoned, test_set_poisoned = split_dataset_by_class(test_data_clean, test_data_poisoned, args.nsample)

        
        # filtered_test_set = remove_attack_label_samples(test_data_clean, test_data_poisoned, args.attack_label)

        filtered_test_set = remove_attack_label_samples(test_set_clean, test_set_poisoned, args.attack_label)
        poisoned_test_loader = DataLoader(dataset=filtered_test_set, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)
        atkmodel = None
    else:
        atkmodel = Autoencoder(spiking_neuron=neuron.IFNode,
                                surrogate_function=surrogate.ATan(),  detach_reset=True)
        atkmodel.to(device)
        atkmodel.load_state_dict(torch.load(f'{args.model_dir}/autoencoder/{args.dataset}-ae-{args.attack_label}.pth'))
        functional.set_step_mode(atkmodel, 'm')

        if torch.cuda.is_available():
            functional.set_backend(atkmodel, 'cupy', instance=neuron.LIFNode)
            cupy.random.seed(42)
    
        filtered_test_set = remove_attack_label_samples(test_set_clean,test_set_clean, args.attack_label)
        # print(len(filtered_test_set))
        poisoned_test_loader = DataLoader(dataset=filtered_test_set, batch_size=poison_args.batch_size, shuffle=False, num_workers=0)
    
    
    ca, asr = evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
    print(f'Initial CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')
    
    if args.algorithm == 'finetuning':
        ca, asr = finetuning(model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, poison_args.trigger_label, device, epochs=args.nstep)      
        algorithm = 'finetuning'  
    elif args.algorithm == 'clamping':
        ca, asr, _ = clamping(args.dataset, model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, poison_args.trigger_label, device, epochs=args.nstep, lr=0.1, c=1e-5, min_c=1e-6, a=1.0, baseline_accuracy=85)
        algorithm = f'clamping-{args.clamp_method}'
    elif args.algorithm == 'self_tuning':
        ca, asr = self_tuning(model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, args, device, epochs=args.nstep)
        algorithm = 'self_tuning'
    elif args.algorithm == 'original':
        ca, asr = evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, args.attack_type, args.attack_label, device)
        algorithm = 'original'
    elif args.algorithm == 'mmbm':
        ca, asr = mmbm(args.dataset, model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, poison_args.trigger_label, device, epochs=args.nstep, lr=0.1, c=1e-5, min_c=1e-6, a=1.2, baseline_accuracy=85)
        algorithm = 'mmbm'
    elif args.algorithm == 'e2e':
        ca, asr = e2e(args.dataset, model, atkmodel, train_loader, clean_test_loader, poisoned_test_loader, args.attack_label, device, args)
        algorithm = 'e2e'    
        
    print(f'CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')

    save_result(args.dataset, args.attack_type, args.attack_label, algorithm, ca, asr)


if __name__ == "__main__":
    main()
