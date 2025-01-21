import argparse
import torch
import numpy as np
from models import get_model
from utils import loss_picker, optimizer_picker
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron
import random
import cupy
from datasets import get_dataset
from torch.utils.data import DataLoader
from utils import train

from poisoned_dataset import PoisonedDataset
from torch.utils.data import Subset



#python get_models.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30 --batch_size 2

#python main.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30 --batch_size 2 --trigger_label 81


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='gesture', help='Dataset to use')
parser.add_argument('--target_label', '-t',type=int, default=8, help='Target label')
parser.add_argument('--target_size', '-s',type=int, default=90, help='Target size')
parser.add_argument('--rest_size', '-r',type=int, default=90, help='Rest size')


parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
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
parser.add_argument('--polarity', default=0, type=int,
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

args = parser.parse_args()






def filter_dataset_by_class(dataset, target_class, rest_size, target_size):
    """
    Filters a given dataset by keeping only the first `target_size` samples for `target_class`
    and the first `rest_size` samples for all other classes.

    Args:
        ref_dataset: A reference PyTorch Dataset object to determine class indices.
                     It is expected that ref_dataset[idx] returns (data, label),
                     where label is either one-hot encoded or can be converted to a class index via argmax().
        dataset: A PyTorch Dataset object to create the filtered subsets from.
        target_class (int): The class index for which to retain `target_size` samples.
        rest_size (int): The number of samples to retain for all other classes.
        target_size (int): The number of samples to retain for the `target_class`.

    Returns:
        filtered_subset (Subset): A subset containing the filtered samples according to the specified sizes.
    """
    # Collect the indices of each class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        # Convert one-hot encoded label to a class index
        label_idx = label.argmax().item() if hasattr(label, 'argmax') else int(label)
        if label_idx not in class_indices:
            class_indices[label_idx] = []
        class_indices[label_idx].append(idx)

    # Select indices for the target class and other classes
    selected_indices = []
    for label_idx, indices in class_indices.items():
        if label_idx == target_class:
            selected_indices.extend(indices[:target_size])  # Take the first `target_size` samples of the target class
        else:
            selected_indices.extend(indices[:rest_size])  # Take the first `rest_size` samples of other classes

    # Create a subset with the selected indices
    filtered_subset = Subset(dataset, selected_indices)

    return filtered_subset



def train_model(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load the model
    model = get_model(args.dataset, args.T)
    functional.set_step_mode(model, 'm')

    if args.cupy:
        functional.set_backend(model, 'cupy', instance=neuron.LIFNode)
        seed = np.uint64(args.seed)
        cupy.random.seed(seed)

    model = model.to(device)

    criterion = loss_picker(args.loss)
    optimizer, scheduler = optimizer_picker(
        args.optim, model.parameters(), args.lr, args.momentum, args.epochs)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    train_data, _ = get_dataset(args.dataset, args.T, args.data_dir)

    train_data = filter_dataset_by_class(train_data, args.target_label, args.rest_size, args.target_size)

    # print("Finished get")
    # train_data = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=args.epsilon,
    #                              pos=args.pos, attack_type=args.type, time_step=args.T,
    #                              trigger_size=args.trigger_size, dataname=args.dataset,
    #                              polarity=args.polarity, n_masks=args.n_masks, least=args.least,
    #                              most_polarity=args.most_polarity, frame_gap=args.frame_gap)
    # print("Finished Poisoned")
    
    
    train_data_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    list_train_loss = []
    list_train_acc = []
    print(f'\n[!] Training the model for {args.epochs} epochs')
    print(f'\n[!] Trainset size is {len(train_data_loader.dataset)}')
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            model, train_data_loader, optimizer, criterion, device, scaler, scheduler)
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        print(f'\n[!] Epoch {epoch + 1}/{args.epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} ')
    return model




model = train_model(args)
torch.save(model.state_dict(), f"models/{args.dataset}-clean-{args.target_size}.pth")