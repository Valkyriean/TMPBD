import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
from spikingjelly.activation_based import functional, neuron
from torch.cuda import amp
import torch.nn.functional as F
from torch.utils.data import Subset


def loss_picker(loss):
    '''
    Select the loss function
    Parameters:
        loss (str): name of the loss function
    Returns:
        loss_function (torch.nn.Module): loss function
    '''
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum, epochs):
    '''
    Select the optimizer
    Parameters:
        optimization (str): name of the optimization method
        param (list): model's parameters to optimize
        lr (float): learning rate
    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    '''
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs)

    return optimizer, lr_scheduler


def train(model, train_loader, optimizer, criterion, device, scaler=None, scheduler=None):
    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = 11

    for frame, label in tqdm(train_loader, disable=True):
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        # If label is not one-hot,
        if len(label.shape) == 1:
            label = F.one_hot(label, n_classes).float()
        if scaler is not None:
            with amp.autocast():
                # Mean is important; (https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/conv_fashion_mnist.html)
                # we need to average the output in the time-step dimension to get the firing rates,
                # and then calculate the loss and accuracy by the firing rates
                out_fr = model(frame).mean(0)
                loss = criterion(out_fr, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

        label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    # model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in tqdm(test_loader, disable=True):
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)
            label = label.to(device)
            # label_onehot = F.one_hot(label, 11).float()
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)

            label = label.argmax(1)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc


def path_name(args):
    """
    Generate the path name based on th experiment arguments. Use a function for
    that to allow checking the existence of the path from different scripts.
    Parameters:
        args (argparse.Namespace): script arguments.
    Returns:
        path (string): The path used to save our experiments
    """
    if args.epsilon == 0.0:
        path = f'clean_{args.dataset}_{args.seed}'
    elif args.type == 'smart' or args.type == 'dynamic':
        path = f'{args.dataset}_{args.type}_{args.epsilon}_{args.trigger_size}_{args.seed}'
    else:
        path = f'{args.dataset}_{args.type}_{args.epsilon}_{args.trigger_size}_{args.pos}_{args.polarity}_{args.seed}'

    path = os.path.join(args.save_path, path)
    return path


def backdoor_model_trainer(model, criterion, optimizer, epochs, poison_trainloader, clean_testloader,
                           poison_testloader, device, scaler=None, scheduler=None):

    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    list_test_loss_backdoor = []
    list_test_acc_backdoor = []

    print(f'\n[!] Training the model for {epochs} epochs')
    print(f'\n[!] Trainset size is {len(poison_trainloader.dataset)},'
          f'Testset size is {len(clean_testloader.dataset)},'
          f'and the poisoned testset size is {len(poison_testloader.dataset)}'
          )

    for epoch in range(epochs):

        train_loss, train_acc = train(
            model, poison_trainloader, optimizer, criterion, device, scaler, scheduler)

        test_loss_clean, test_acc_clean = evaluate(
            model, clean_testloader, criterion, device)

        test_loss_backdoor, test_acc_backdoor = evaluate(
            model, poison_testloader, criterion, device)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)
        list_test_loss_backdoor.append(test_loss_backdoor)
        list_test_acc_backdoor.append(test_acc_backdoor)

        print(f'\n[!] Epoch {epoch + 1}/{epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Test acc: {test_acc_clean:.4f} '
              f'Test acc backdoor: {test_acc_backdoor:.4f}'
              )

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor


def plot_accuracy_combined(name, list_train_acc, list_test_acc, list_test_acc_backdoor):
    '''
    Plot the accuracy of the model in the main and backdoor test set
    Parameters:
        name (str): name of the figure
        list_train_acc (list): list of train accuracy for each epoch
        list_test_acc (list): list of test accuracy for each epoch
        list_test_acc_backdoor (list): list of test accuracy for poisoned test dataset
    Returns:
        None
    '''

    sns.set()

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(name)

    ax[0].set_title('Training accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].plot(list_train_acc)

    ax[1].set_title('Test accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(list_test_acc)

    ax[2].set_title('Test accuracy backdoor')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Accuracy')
    ax[2].plot(list_test_acc_backdoor)

    plt.savefig(f'{name}/accuracy.png',  bbox_inches='tight')
    # Also saving as pdf for using the plot in the paper
    plt.savefig(f'{name}/accuracy.pdf',  bbox_inches='tight')


def save_experiments(args, train_acc, train_loss, test_acc_clean, test_loss_clean, test_acc_backdoor,
                     test_loss_backdoor, model):

    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/results.csv'.format(args.save_path)
    header = ['dataset', 'least', 'most_polarity', 'seed', 'epsilon', 'pos',
              'polarity', 'trigger_size', 'trigger_label',
              'loss', 'optimizer', 'batch_size', 'type', 'epochs', 
              'train_acc', 'test_acc_clean', 'test_acc_backdoor', 'frame_gaps']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, args.least, args.most_polarity, args.seed, args.epsilon, args.pos,
                         args.polarity, args.trigger_size, args.trigger_label,
                         train_loss[-1], args.optim, args.batch_size, args.type, args.epochs,
                         train_acc[-1], test_acc_clean[-1], test_acc_backdoor[-1], args.frame_gap])

    # Create a folder for the experiment, named after the experiment
    path = path_name(args)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the info in a file
    with open(f'{path}/args.txt', 'w') as f:
        f.write(str(args))

    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'list_test_loss_backdoor': test_loss_backdoor,
        'list_test_acc_backdoor': test_acc_backdoor,
    }, f'{path}/data.pt')

    torch.save(model, f'{path}/model.pth')

    plot_accuracy_combined(path, train_acc,
                           test_acc_clean, test_acc_backdoor)
    print('[!] Model and results saved successfully!')



def get_data_meta(dataset, npara):
    if dataset == 'gesture':
        input_dim = ( npara, 2, 128, 128)
        NC = 11
        penultimate_neuron = neuron.LIFNode

    elif dataset == 'cifar10':
        input_dim = (npara, 2, 128, 128)
        NC = 10
        penultimate_neuron = neuron.LIFNode

    elif dataset == 'mnist':
        input_dim = (npara, 2, 34, 34)
        NC = 10
        penultimate_neuron = neuron.IFNode
    elif dataset == 'caltech':
        input_dim = (npara, 2, 180, 180)
        NC = 101
        penultimate_neuron = neuron.LIFNode
    return input_dim, NC, penultimate_neuron

def split_dataset_by_class(ref_dateset, dataset, n=50):
    """
    Splits the given dataset by class, taking the first n samples from each class 
    and separating the remaining samples.

    Args:
        dataset: A PyTorch Dataset object. It is expected that dataset[idx] returns (data, label),
                 where label is either one-hot encoded or can be converted to a class index via argmax().
        n (int): The number of samples to select per class. Default is 50.

    Returns:
        selected_subset (Subset): A subset containing the first n samples per class.
        remaining_subset (Subset): A subset containing all other samples.
    """

    # Collect the indices of each class
    class_indices = {}
    for idx in range(len(ref_dateset)):
        _, label = ref_dateset[idx]
        # Convert one-hot encoded label to a class index
        label_idx = label.argmax().item()
        if label_idx not in class_indices:
            class_indices[label_idx] = []
        class_indices[label_idx].append(idx)

    # Select the first n samples for each class
    selected_indices = []
    for label_idx, indices in class_indices.items():
        selected_indices.extend(indices[:n])  # take the first n samples of each class

    # Compute the indices of the remaining samples
    all_indices = set(range(len(ref_dateset)))
    selected_indices_set = set(selected_indices)
    remaining_indices = list(all_indices - selected_indices_set)

    # Create subsets
    selected_subset = Subset(dataset, selected_indices)
    remaining_subset = Subset(dataset, remaining_indices)

    return selected_subset, remaining_subset


# def evaluate(model, test_loader, criterion, device):
#     # model.eval()
#     test_loss = 0
#     test_acc = 0
#     test_samples = 0
#     with torch.no_grad():
#         for frame, label in test_loader:
#             print(test_samples)
#             frame = frame.to(device)
#             # [N, T, C, H, W] -> [T, N, C, H, W]
#             frame = frame.transpose(0, 1)
#             label = label.to(device)
#             out_fr = model(frame).mean(0)
#             # loss = criterion(out_fr, label)

#             label = label.argmax(1)
#             test_samples += label.numel()
#             # test_loss += loss.item() * label.numel()
#             test_acc += (out_fr.argmax(1) == label).float().sum().item()

#             functional.reset_net(model)

#     # test_loss /= test_samples
#     test_acc /= test_samples

#     return test_acc

def clip_image(image, noise, eps):
    '''
    Clip the noise so its l infinity norm is less than eps
    noise shape: [T, N, C, H, W]
    image shape: [T, N, C, H, W]
    '''
    noise = noise * eps
    return noise + image

# def dynamic_devaluate(model, atkmodel, test_loader, trigger_label, device):
    
#     try:
#         n_classes = len(test_loader.dataset.classes)
#     except:
#         n_classes = 10

#     # crop = None
#     # TODO crop caltech data
#     # if args.dataset == 'caltech':
#     #     n_classes = 101
#     #     crop = transforms.CenterCrop((180, 180))

#     bk_label_one_hot = F.one_hot(torch.tensor(
#         0).long(), n_classes).float()

#     # atkmodel.eval()
#     # model.eval()
#     test_bk_acc = 0
#     test_samples = 0
#     with torch.no_grad():
#         for frame, label in test_loader:
#             frame = frame.to(device)
#             # [N, T, C, H, W] -> [T, N, C, H, W]
#             frame = frame.transpose(0, 1)
#             label = label.to(device)
#             # bk_label = bk_label_one_hot.repeat(len(label), 1).to(device)
#             # if crop is not None:
#             #     frame = crop(frame)
                
#             # output = model(frame).mean(0)
#             # label = label.argmax(1)
#             test_samples += label.numel()
#             # test_acc += (output.argmax(1) == label).float().sum().item()
#             # functional.reset_net(model)

#             noise = atkmodel(frame)
#             atkdata = clip_image(frame, noise, 0.01)
#             bk_output = model(atkdata).mean(0)
#             # loss = criterion(out_fr, bk_label)
#             test_bk_acc += (bk_output.argmax(1) ==
#                             trigger_label).float().sum().item()
#             functional.reset_net(atkmodel)
#             functional.reset_net(model)

#     # test_loss /= test_samples
    
#     test_bk_acc /= test_samples
#     return test_bk_acc


def dynamic_evaluate(model, atkmodel, test_loader, trigger_label, device):
    
    try:
        n_classes = len(test_loader.dataset.classes)
    except:
        n_classes = 10

    bk_label_one_hot = F.one_hot(torch.tensor(
        0).long(), n_classes).float()

    atkmodel.eval()
    test_acc = 0
    test_bk_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)
            label = label.to(device)
            bk_label = bk_label_one_hot.repeat(len(label), 1).to(device)
            # if crop is not None:
            #     frame = crop(frame)

            output = model(frame).mean(0)
            label = label.argmax(1)
            test_samples += label.numel()
            test_acc += (output.argmax(1) == label).float().sum().item()
            functional.reset_net(model)

            noise = atkmodel(frame)
            atkdata = clip_image(frame, noise, 0.01)
            bk_output = model(atkdata).mean(0)
            # loss = criterion(out_fr, bk_label)
            test_bk_acc += (bk_output.argmax(1) ==
                            trigger_label).float().sum().item()
            functional.reset_net(atkmodel)
            functional.reset_net(model)

    # test_loss /= test_samples
    
    test_acc /= test_samples
    test_bk_acc /= test_samples
    return test_acc, test_bk_acc


def remove_attack_label_samples(ref_set, test_set, attack_label):
    """
    Remove samples from test_set whose label equals attack_label.
    
    Args:
        test_set: A PyTorch Dataset object. 
                  test_set[idx] is expected to return (data, label).
                  label can be an integer or a one-hot encoded tensor.
        attack_label (int): The label to remove from the dataset.

    Returns:
        filtered_test_set (Subset): A subset of test_set that excludes samples with attack_label.
    """
    filtered_indices = []

    for idx in range(len(ref_set)):
        data, label = ref_set[idx]
        
        # If label is one-hot encoded, convert to integer index
        if hasattr(label, 'argmax'):
            label = label.argmax().item()

        if label != attack_label:
            filtered_indices.append(idx)

    filtered_test_set = Subset(test_set, filtered_indices)
    return filtered_test_set