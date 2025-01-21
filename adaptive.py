import argparse
import csv
import os
import torch
import numpy as np
from models import get_model, Autoencoder
from utils import loss_picker, optimizer_picker
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
import random
import cupy
from datasets import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from poisoned_dataset import PoisonedDataset
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_data_meta, split_dataset_by_class, evaluate, dynamic_evaluate, remove_attack_label_samples, clip_image


#python get_models.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30 --batch_size 2

#python main.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30 --batch_size 2 --trigger_label 81


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='gesture', help='Dataset to use')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
parser.add_argument('--T', default=16, type=int,
                    help='simulating time-steps')
parser.add_argument('--amp', action='store_true',
                    help='Use automatic mixed precision training')
parser.add_argument('--cupy', action='store_true', help='Use cupy')
parser.add_argument('--loss', type=str, default='mse',
                    help='Loss function', choices=['mse', 'cross'])
parser.add_argument('--optim', type=str, default='adam',
                    help='Optimizer', choices=['adam', 'sgd'])
# Trigger related parameters
parser.add_argument('--trigger_label', default=0, type=int,
                    help='The index of the trigger label')
parser.add_argument('--polarity', default=1, type=int,
                    help='The polarity of the trigger', choices=[0, 1, 2, 3])
parser.add_argument('--trigger_size', default=0.1,
                    type=float, help='The size of the trigger as the percentage of the image size')
parser.add_argument('--epsilon', default=0.1, type=float,
                    help='The percentage of poisoned data')
parser.add_argument('--pos', default='top-left', type=str,
                    help='The position of the trigger', choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'])
parser.add_argument('--type', default='static', type=str,
                    help='The type of the trigger', choices=['static', 'moving', 'smart','hash','shr','shs','rs','blink'])
parser.add_argument('--n_masks', default=2, type=int,
                    help='The number of masks. Only if the trigger type is smart')
parser.add_argument('--least', action='store_true',
                    help='Use least active area for smart attack')
parser.add_argument('--most_polarity', action='store_true',
                    help='Use most active polarity in the area for smart attack')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
# Other
parser.add_argument('--data_dir', type=str,
                    default='data', help='Data directory')
parser.add_argument('--save_path', type=str,
                    default='experiments', help='Path to save the experiments')
parser.add_argument('--model_path', type=str, default=None,
                    help='Use a pretrained model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--frame_gap', default=1, type=int,
                    help='Inject trigger into every x frames')
parser.add_argument('--weight', default=0, type=float,
                    help='TMP penalty weight')


args = parser.parse_args()










def evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, attack_type, attack_label, device):
    criterion = nn.MSELoss()
    if attack_type == "dynamic":
        _, ca = evaluate(model, clean_test_loader, criterion, device)
        _, asr = dynamic_evaluate(model, atkmodel, poisoned_test_loader,attack_label, device)
    else:
        _, ca = evaluate(model, clean_test_loader, criterion, device)
        _, asr = evaluate(model, poisoned_test_loader, criterion, device)
    return ca, asr




def train(model, train_loader, optimizer, criterion, device, args, scaler=None, scheduler=None, tmp_penalty_weight=1.0):
    """
    Train the model with an additional penalty term in the loss for target label TMP.
    
    Args:
        tmp_penalty_weight (float): Weight for the TMP penalty term.
    """
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = 11

    


    for frame, label in tqdm(train_loader, disable=True):
        labels = args.trigger_label * torch.ones((len(frame),), dtype=torch.long).to(device)
        # print(labels.shape)
        onehot_label = F.one_hot(labels, num_classes=n_classes)
        # print(onehot_label.shape)
        functional.reset_net(model)
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        # If label is not one-hot,
        if len(label.shape) == 1:
            label = F.one_hot(label, n_classes).float()
        


        
        # Calculate TMP penalty
        v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=model, instance=neuron.LIFNode)
        out_fr = model(frame).mean(0)
        # Calculate base loss
        base_loss = criterion(out_fr, label)
        v = v_seq_monitor.records[-1].mean(0)
        v_max = torch.max(v_seq_monitor.records[-1], dim=0)[0]
        v_seq_monitor.clear_recorded_data()
        v_seq_monitor.remove_hooks()
        del v_seq_monitor
        v_pool = torch.avg_pool1d(v, kernel_size=10, stride=10)
        # print(v_pool.shape)

        # tmp = torch.max(out_fr, dim=1)[0]  # Maximum activation per class
        target_label_tmp = v_pool[args.trigger_label]  # TMP for target labels
        # penalty = target_label_tmp.mean()  # Penalty for overlarge TMP
        penalty = torch.sum((v_pool * onehot_label))
        # print(penalty)
        # Total loss
        loss = base_loss + tmp_penalty_weight * penalty
        # print(f"Loss: {loss.item()}, Base loss: {base_loss.item()}, Penalty: {penalty}")
        
        loss.backward()
        optimizer.step()

        # Update metrics
        label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        # Reset the model state
        functional.reset_net(model)

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc





def train_model(args, device):

    # Set the device

    # Load the model
    model = get_model(args.dataset, args.T)
    functional.set_step_mode(model, 'm')

    if args.cupy:
        functional.set_backend(model, 'cupy', instance=neuron.LIFNode)
        seed = np.uint64(args.seed)
        cupy.random.seed(seed)

    model = model.to(device)
    for m in model.modules():
        if isinstance(m, neuron.LIFNode):
            m.store_v_seq = True
    criterion = loss_picker(args.loss)
    optimizer, scheduler = optimizer_picker(
        args.optim, model.parameters(), args.lr, args.momentum, args.epochs)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)
    test_data = None
    print("Finished get")
    train_data = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=args.epsilon,
                                 pos=args.pos, attack_type=args.type, time_step=args.T,
                                 trigger_size=args.trigger_size, dataname=args.dataset,
                                 polarity=args.polarity, n_masks=args.n_masks, least=args.least,
                                 most_polarity=args.most_polarity, frame_gap=args.frame_gap)
    print("Finished Poisoned")
    train_data_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    list_train_loss = []
    list_train_acc = []
    print(f'\n[!] Training the model for {args.epochs} epochs')
    print(f'\n[!] Trainset size is {len(train_data_loader.dataset)}')
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            model, train_data_loader, optimizer, criterion, device, args, scaler, scheduler, args.weight)
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        print(f'\n[!] Epoch {epoch + 1}/{args.epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} ')
    return model



def model_testing(model, args, device):
    _, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    test_data_clean = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=0,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity, frame_gap = args.frame_gap)


    # train_loader = DataLoader(dataset=test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=0)
    clean_test_loader = DataLoader(dataset=test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=0)

    
    if args.type != "dynamic":
        if args.type == "clean":
            test_data_poisoned = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=1,
                                            pos=args.pos, attack_type="static", time_step=args.T,
                                            trigger_size=args.trigger_size, dataname=args.dataset,
                                            polarity=args.polarity, n_masks=args.n_masks, least=args.least, 
                                            most_polarity=args.most_polarity, frame_gap = args.frame_gap)

        else:
            test_data_poisoned = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=1,
                                            pos=args.pos, attack_type=args.type, time_step=args.T,
                                            trigger_size=args.trigger_size, dataname=args.dataset,
                                            polarity=args.polarity, n_masks=args.n_masks, least=args.least, 
                                            most_polarity=args.most_polarity, frame_gap = args.frame_gap)


        filtered_test_set = remove_attack_label_samples(test_data_clean, test_data_poisoned, args.trigger_label)
        poisoned_test_loader = DataLoader(dataset=filtered_test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        atkmodel = None
    else:
        atkmodel = Autoencoder(spiking_neuron=neuron.IFNode,
                                surrogate_function=surrogate.ATan(),  detach_reset=True)
        atkmodel.to(device)
        atkmodel.load_state_dict(torch.load(f'{args.model_dir}/autoencoder/{args.dataset}-ae-{args.trigger_label}.pth'))
        functional.set_step_mode(atkmodel, 'm')

        if torch.cuda.is_available():
            functional.set_backend(atkmodel, 'cupy', instance=neuron.LIFNode)
            cupy.random.seed(42)
    
        filtered_test_set = remove_attack_label_samples(test_data_clean,test_data_clean, args.trigger_label)
        poisoned_test_loader = DataLoader(dataset=filtered_test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    
    ca, asr = evaluate_ca_asr(model, atkmodel, clean_test_loader, poisoned_test_loader, args.type, args.trigger_label, device)
    print(f'CA: {ca * 100:.2f}%, ASR: {asr * 100:.2f}%')
    return ca, asr
    
    
def save_result(dataset, attack_type, id, weight, ca, asr):
    save_path = "experiments"
    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/adaptive.csv'.format(save_path)
    header = ['Dataset', 'Attack Type', 'Model ID', 'Weight', 'CA', 'ASR']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, attack_type, id, weight, f"{ca * 100:.2f}",f"{asr * 100:.2f}"])

    
    
def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = train_model(args, device)
    torch.save(model.state_dict(), f"models/{args.dataset}-{args.type}-adp-{args.trigger_label}-{args.weight}.pth")
    
    # model = get_model(args.dataset, 16)
    # model = model.to(device)
    # model_path = f'models/gesture-static-0.pth'
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
    # functional.set_step_mode(model, 'm')

    
    ca, asr = model_testing(model, args, device)

    save_result(args.dataset, args.type, args.trigger_label, args.weight, ca, asr)
if __name__ == "__main__":
    main()
