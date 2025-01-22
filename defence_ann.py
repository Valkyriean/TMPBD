import torch
from tqdm import tqdm
import numpy as np
from spikingjelly.activation_based import functional, neuron
from torch.utils.data import DataLoader, Subset
import cupy
import argparse
from models import get_model
from datasets import get_dataset
from poisoned_dataset import PoisonedDataset

import os, csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from spikingjelly.activation_based import functional
from utils import get_data_meta


parser = argparse.ArgumentParser(description='ANNs Defence Adoption on SNNs')
parser.add_argument('--model_dir',type=str, default='models', help='model path')
parser.add_argument('--dataset', '-d', type=str, default='gesture', choices=["gesture", "cifar10", "caltech", "mnist"], help='dataset name')
parser.add_argument('--attack_type', '-t', type=str, default='static', choices=["clean", "static", "dynamic"], help='attack type')
parser.add_argument('--attack_label', '-l',type=int, default=0, help='attack label')
parser.add_argument('--model_name', type=str, default=None, help='model name')


parser.add_argument('--nstep', type=int, default=5000, help='max optimisation epoches')
parser.add_argument('--npara', type=int, default=1, help='number of parallel initial states')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--early_stop', action='store_true',help='to use early stop')
parser.add_argument('--save_plot', action='store_true',help='save the learning graph')

parser.add_argument('--target', type=str, default="v_avg",choices=["fr", "v_avg", "v_max", "h"] ,help='which variable targeted to be optimised')
parser.add_argument('--algorithm', type=str, default="abs",choices=["abs", "nc"] ,help='which algorithm to use')


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


args1 = Args()


np.random.seed(42)
torch.manual_seed(42)
# random.seed(42)



def stimulate_neuron(model, input_shape, target_neuron, device, timesteps=16):
    """
    Stimulate a specific neuron by generating synthetic inputs and recording its spiking behavior.
    """
    model.eval()
    # Create a leaf tensor with `requires_grad=True`
    synthetic_input = torch.rand((timesteps, *input_shape), device=device) * 0.1
    synthetic_input = synthetic_input.clone().detach().requires_grad_(True)  # Ensure it's a leaf tensor

    optimizer = torch.optim.Adam([synthetic_input], lr=0.01)

    for _ in range(100):  # Iterative optimization to stimulate the target neuron
        functional.reset_net(model)  # Reset SNN state
        out = model(synthetic_input)  # Forward pass

        # Ensure the target activation is differentiable
        target_activation = out.mean(0)[:, target_neuron]  # Mean spike rate for the target neuron
        target_activation.requires_grad = True
        loss = -target_activation.mean()  # Maximize target neuron activation
        # print(target_activation.requires_grad)

        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # print(loss.requires_grad)
        loss.backward()  # Backpropagation
        optimizer.step()

        # Clamp the synthetic input to valid range
        synthetic_input.data.clamp_(0, 1)

    return synthetic_input.detach()



def analyze_neurons(model, input_shape, num_classes, device, timesteps=10):
    """
    Analyze neurons to detect suspicious activity.
    """
    model.eval()
    avg_activation = np.zeros((1, num_classes))  # Ensure the shape matches your needs

    for neuron_id in range(num_classes):
        synthetic_input = stimulate_neuron(model, input_shape, neuron_id, device, timesteps)
        out = model(synthetic_input).mean(dim=0).detach().cpu().numpy()
        avg_activation[0, neuron_id] = out.mean()  # Example aggregation, adjust as needed

    # Flatten avg_activation
    avg_activation = avg_activation.flatten()  # Convert to 1D array
    print("avg_activation shape after flattening:", avg_activation.shape)

    # Detect suspicious neurons
    suspicious_neurons = []
    threshold = np.percentile(avg_activation, 95)  # 95th percentile
    for neuron_id in range(num_classes):
        if avg_activation[neuron_id] > threshold:  # Compare scalar values
            suspicious_neurons.append(neuron_id)

    return suspicious_neurons, avg_activation



def save_result(dataset, attack_type, attack_label, suspicious_neurons, suspicious_neuron_score, result, correct):
    save_path = "experiments"
    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/ann_results.csv'.format(save_path)
    header = ['Dataset', 'Attack Type', 'Model ID', 'suspicious_neurons', 'suspicious_neuron_score', 'result', 'Correctness']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, attack_type, attack_label, suspicious_neurons, suspicious_neuron_score, result, correct])
        
    

# Determine if a model is backdoored
def detect_backdoor_abs(model, test_loader, device, input_shape, num_classes, threshold=0.5):
    """
    Perform ABS-based detection and classify the model as backdoored or clean.
    """
    suspicious_neurons, activations = analyze_neurons(model, input_shape, num_classes, device)

    print(f"Suspicious neurons detected: {suspicious_neurons}")

    # Step 1: Generate activations for clean and backdoor inputs
    print("[!] Evaluating model with clean and backdoor inputs...")
    clean_activations = []
    backdoor_activations = []

    for frame, label in tqdm(test_loader, disable=False):
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)

        # Process clean inputs
        functional.reset_net(model)
        clean_output = model(frame).mean(0)
        clean_activations.append(clean_output.cpu().detach().numpy())
        neuron_id = label.argmax().item() 

        # Process backdoor inputs (simulate)
        synthetic_input = stimulate_neuron(model, input_shape, neuron_id, device)

        backdoor_frame = frame + synthetic_input
        functional.reset_net(model)
        backdoor_output = model(backdoor_frame).mean(0)
        backdoor_activations.append(backdoor_output.cpu().detach().numpy())
    # Step 2: Compare activations
    clean_activations = np.array(clean_activations)
    backdoor_activations = np.array(backdoor_activations)

    print(f"clean_activations.shape: {clean_activations.shape}")
    print(f"backdoor_activations.shape: {backdoor_activations.shape}")

    activation_diff = backdoor_activations.mean(axis=0).mean(axis=0) - clean_activations.mean(axis=0).mean(axis=0)
    
    print(f"activation_diff.shape: {activation_diff.shape}")
    valid_neurons = [n for n in suspicious_neurons if n < activation_diff.shape[0]]
    
    
    if not valid_neurons:
        print("[!] No valid suspicious neurons detected. Model is classified as CLEAN.")
        result = "No valid suspicious neurons"
        if args.attack_type == 'clean':
            correct = True
        else:
            correct = False
        save_result(args.dataset, args.attack_type, args.attack_label, suspicious_neurons, None, result, correct)
        return False

    # Compute suspicious neuron score
    suspicious_neuron_score = activation_diff[valid_neurons].mean()
    print(f"[!] Suspicious neuron score: {suspicious_neuron_score}")

    

    # Threshold for backdoor detection
    threshold = 0.5  # Example threshold
    if suspicious_neuron_score > threshold:
        print("[!] Model is classified as BACKDOORED.")
        result = "Classified as BACKDOORED"
        if args.attack_type != 'clean':
            correct = True
            if args.attack_label in suspicious_neurons:
                result = "Classified as BACKDOORED and Attack Label Detected"
        else:
            correct = False
        save_result(args.dataset, args.attack_type, args.attack_label, suspicious_neurons, suspicious_neuron_score, result, correct)
        return True
    else:
        print("[!] Model is classified as CLEAN.")
        result = "Classified as CLEAN"
        if args.attack_type == 'clean':
            correct = True
        else:
            correct = False
        save_result(args.dataset, args.attack_type, args.attack_label, suspicious_neurons, suspicious_neuron_score, result, correct)
        return False
def detect_backdoor_NC(
    model,
    test_loader,
    device,
    input_shape,
    num_classes,
    max_iterations=200,
    regularization_type='l1',
    reg_lambda=1e-4,
    lr=0.1,
    anomaly_threshold=2.0,
    print_freq=10  # New parameter: controls how many iterations before printing loss
):
    """
    Based on the Neural Cleanse approach for backdoor detection (suitable for SNN + Gesture DVS data):
    1. For each possible target label, we seek a trigger (mask+pattern) that minimizes the classification loss of inputs on that label.
    2. Compare the trigger norms for all target labels, and use anomaly detection methods (MAD, median-based, etc.) to decide whether a backdoor exists.
    3. If a backdoor exists, the label corresponding to the smallest trigger norm is typically considered the backdoor target.
    
    * New: output each part of the loss (ce_loss + reg_loss) in the middle.
    """
    model.eval()

    timesteps = 16
    trigger_norms = []

    # ---------- Step 1: Optimize the minimal perturbation trigger for each target_label ----------
    for target_label in range(num_classes):
        # The shape of "trigger" matches (T, C, H, W)
        # trigger = torch.zeros((timesteps, *input_shape), dtype=torch.float32, device=device, requires_grad=True)
        trigger = torch.rand((timesteps, *input_shape), device=device) * 0.1
        trigger = trigger.clone().detach().requires_grad_(True)  # Ensure it's a leaf tensor
        
        optimizer = torch.optim.Adam([trigger], lr=lr)

        # print(f"\n[NC] >>> Optimizing trigger for target_label = {target_label}")
        for iteration in range(max_iterations):
            # To reduce printing, you may consider whether to place the test_loader part in an outer loop
            # Here, for demonstration, we optimize on all data within the inner loop
            # If the dataset is large, you could also sample only a few batches
            total_ce_loss = 0.0
            total_reg_loss = 0.0
            batch_count = 0

            for frame, label in test_loader:
                frame = frame.to(device)
                label = label.to(device)

                # [N, T, C, H, W] -> [T, N, C, H, W]
                frame = frame.transpose(0, 1)

                functional.reset_net(model)
                poisoned_data = frame + trigger
            
                outputs = model(poisoned_data)  # [T, N, num_classes]
                outputs = outputs.mean(dim=0)   # [N, num_classes]

                # Construct the target so that all samples are classified as target_label
                target = torch.zeros_like(outputs, device=device)
                target[:, target_label] = 1.0

                ce_loss = F.binary_cross_entropy_with_logits(outputs, target)
                if regularization_type == 'l1':
                    reg_loss = torch.norm(trigger, p=1)
                else:
                    reg_loss = torch.norm(trigger, p=2)

                total_loss = ce_loss + reg_lambda * reg_loss
                optimizer.zero_grad()
                total_loss.backward()
                # print(f"trigger grad: {trigger.grad.sum()}")
                optimizer.step()

                # Record the loss
                total_ce_loss += ce_loss.item()
                total_reg_loss += reg_loss.item()
                batch_count += 1

            # You can clamp the trigger here
            # trigger.data.clamp_(0, 1)

            # Print the average ce_loss and reg_loss every print_freq iterations
            # if (iteration + 1) % print_freq == 0:
            #     avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0
            #     avg_reg_loss = total_reg_loss / batch_count if batch_count > 0 else 0
            #     print(f"[target_label={target_label}] Iter {iteration+1}/{max_iterations} - "
            #           f"Avg CE Loss: {avg_ce_loss:.6f}, Avg Reg Loss: {avg_reg_loss:.6f}, "
            #           f"Total Loss: {(avg_ce_loss + reg_lambda * avg_reg_loss):.6f}")

        # Compute the final norm of the trigger
        if regularization_type == 'l1':
            trigger_norm = torch.norm(trigger.detach(), p=1).item()
        else:
            trigger_norm = torch.norm(trigger.detach(), p=2).item()

        trigger_norms.append(trigger_norm)

    # ---------- Step 2: Use trigger_norms for anomaly detection ----------
    trigger_norms = np.array(trigger_norms)
    median = np.median(trigger_norms)
    mad = np.median(np.abs(trigger_norms - median))
    mad = mad if mad != 0 else 1e-6

    min_norm = trigger_norms.min()
    if (median - min_norm) / mad > anomaly_threshold:
        is_backdoored = True
    else:
        is_backdoored = False

    # ---------- Step 3: If backdoored, find the most suspicious target_label ----------
    suspected_target_label = int(trigger_norms.argmin())

    if is_backdoored:
        print(f"[NC] Model is classified as BACKDOORED. Suspected target label = {suspected_target_label}.")
    else:
        print("[NC] Model is classified as CLEAN.")
    # pseudo code
    save_result(
        dataset=args.dataset,
        attack_type=args.attack_type,
        attack_label=args.attack_label,
        suspicious_neurons=suspected_target_label,
        suspicious_neuron_score=is_backdoored,
        result=f"BACKDOORED with target_label={suspected_target_label}" if is_backdoored else "CLEAN",
        correct=((args.attack_type == 'clean' and is_backdoored==False) or (args.attack_type != 'clean' and is_backdoored))  # Example only
    )

    return is_backdoored, suspected_target_label







    
# Main Detection Script
def main():
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args.dataset, args1.T)
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

    # print(model)
    # Data preparation
    
    input_shape, num_classes, _ = get_data_meta(args.dataset, args.npara)

     # Get the dataset
    _, test_data = get_dataset(args1.dataset, args1.T, args1.data_dir)




    test_data_ori = PoisonedDataset(test_data, args1.trigger_label, mode='test', epsilon=0,
                                    pos=args1.pos, attack_type=args1.type, time_step=args1.T,
                                    trigger_size=args1.trigger_size, dataname=args1.dataset,
                                    polarity=args1.polarity, n_masks=args1.n_masks, least=args1.least, most_polarity=args1.most_polarity, frame_gap = args1.frame_gap)

    # data in [16, 1, 2, 128, 128], where 16 is the time steps, 1 is the batch size, 2 is the channel, 128 is the height and 128 is the width
    # gesture data has 11 classes
    
    # Extracting 50 Samples per Class from the Test Set
    class_indices = {}
    for idx in range(len(test_data_ori)):
        _, label = test_data_ori[idx]
        label_idx = label.argmax().item()  # Convert one-hot to class index
        if label_idx not in class_indices:
            class_indices[label_idx] = []
        class_indices[label_idx].append(idx)


    selected_indices = []
    for label_idx in class_indices:
        indices = class_indices[label_idx][:50]  # Take the first 50 samples
    selected_indices.extend(indices)

    selected_test_data = Subset(test_data_ori, selected_indices)

    test_loader = DataLoader(dataset=selected_test_data, batch_size=args1.batch_size, shuffle=False, num_workers=0)
    # Define your data loader (clean and backdoor inputs)

    # Perform ABS detection
    if args.algorithm == 'abs':
        is_backdoored = detect_backdoor_abs(model, test_loader, device, input_shape, num_classes)
    elif args.algorithm == 'nc':
        is_backdoored = detect_backdoor_NC(model, test_loader, device, input_shape, num_classes)
    # Output final decision
    print(f"Final Model Classification: {'BACKDOORED' if is_backdoored else 'CLEAN'}")
    

if __name__ == "__main__":
    main()
