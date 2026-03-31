import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_dataloaders
from model import MNIST_CNN, get_cifar10_model
from attacks import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted


# 1. Train and Eval Models
def train_and_evaluate_mnist(model, train_loader, test_loader, device, epochs=3):
    save_path = "mnist_cnn.pth"
    criterion = nn.CrossEntropyLoss()
    
    # load saved model if exists
    if os.path.exists(save_path):
        print("Load saved MNIST model...")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    else:
        print("Start MNIST training...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), save_path)

    # check clean accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"MNIST Clean Test Acc: {100 * correct / total:.2f}%\n")
    return model

def evaluate_cifar10(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # use 1000 samples for fast test
        for i, (images, labels) in enumerate(test_loader):
            if i >= 1000: break
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"CIFAR-10 Clean Test Acc: {100 * correct / total:.2f}%\n")
    return model


# 2. Attack and Plot Logic
def visualize_samples(samples, dataset_name, attack_name, eps):
    # save 5 adv images
    if len(samples) == 0: return
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(8, 2 * len(samples)))
    fig.suptitle(f"{dataset_name} - {attack_name} (eps={eps})", fontsize=16)
    
    for i, (clean_img, adv_img, clean_pred, adv_pred) in enumerate(samples):
        # tensor to numpy [H, W, C]
        c_img = clean_img.squeeze().cpu().numpy()
        a_img = adv_img.squeeze().cpu().numpy()
        
        if c_img.ndim == 3: # for CIFAR-10
            c_img = np.transpose(c_img, (1, 2, 0))
            a_img = np.transpose(a_img, (1, 2, 0))
            
        noise = a_img - c_img
        # make noise visible (0~1)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-5)
        
        # original
        axes[i, 0].imshow(c_img, cmap='gray' if c_img.ndim==2 else None)
        axes[i, 0].set_title(f"Org: {clean_pred}")
        axes[i, 0].axis('off')
        
        # adversarial
        axes[i, 1].imshow(a_img, cmap='gray' if c_img.ndim==2 else None)
        axes[i, 1].set_title(f"Adv: {adv_pred}")
        axes[i, 1].axis('off')
        
        # noise
        axes[i, 2].imshow(noise, cmap='gray' if c_img.ndim==2 else None)
        axes[i, 2].set_title("Noise")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_{attack_name}_eps{eps}.png")
    plt.close()

def run_attack_eval(model, test_loader, attack_fn, is_targeted, eps, device, dataset_name, attack_name, num_samples=100, save_vis=False):
    # run attack and calc acc
    model.eval()
    success_count = 0
    total_count = 0
    saved_samples = []

    for images, labels in test_loader:
        if total_count >= num_samples: break
        
        images, labels = images.to(device), labels.to(device)
        
        # skip if org pred is wrong
        clean_outputs = model(images)
        clean_pred = clean_outputs.argmax(dim=1).item()
        if clean_pred != labels.item():
            continue
            
        # set target = label + 1
        if is_targeted:
            target = (labels + 1) % 10 
            adv_images = attack_fn(model, images, target, eps)
        else:
            adv_images = attack_fn(model, images, labels, eps)
            
        # check adv pred
        adv_outputs = model(adv_images)
        adv_pred = adv_outputs.argmax(dim=1).item()
        
        # check if attack success
        is_success = False
        if is_targeted and adv_pred == target.item():
            is_success = True
        elif not is_targeted and adv_pred != labels.item():
            is_success = True
            
        if is_success:
            success_count += 1
            if len(saved_samples) < 5: 
                saved_samples.append((images[0].detach(), adv_images[0].detach(), clean_pred, adv_pred))
                
        total_count += 1

    sr = (success_count / total_count) * 100 if total_count > 0 else 0
    
    if save_vis and len(saved_samples) > 0:
        visualize_samples(saved_samples, dataset_name, attack_name, eps)
        
    return sr

# ==========================================
# 3. Main
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs("results", exist_ok=True)

    # load data
    mnist_train_loader, mnist_test_loader, cifar_test_loader = get_dataloaders(batch_size=64)
    
    # load models
    mnist_model = MNIST_CNN().to(device)
    mnist_model = train_and_evaluate_mnist(mnist_model, mnist_train_loader, mnist_test_loader, device)
    
    cifar_model = get_cifar10_model().to(device)
    cifar_model = evaluate_cifar10(cifar_model, cifar_test_loader, device)

    # eps list
    epsilons = [0.05, 0.1, 0.2, 0.3]
    
    # wrap pgd to match fgsm params
    pgd_t_wrap = lambda m, x, y, e: pgd_targeted(m, x, y, k=10, eps=e, eps_step=e/4)
    pgd_u_wrap = lambda m, x, y, e: pgd_untargeted(m, x, y, k=10, eps=e, eps_step=e/4)

    attacks = [
        ("FGSM Targeted", fgsm_targeted, True),
        ("FGSM Untargeted", fgsm_untargeted, False),
        ("PGD Targeted", pgd_t_wrap, True),
        ("PGD Untargeted", pgd_u_wrap, False)
    ]

    datasets = [
        ("MNIST", mnist_model, mnist_test_loader),
        ("CIFAR10", cifar_model, cifar_test_loader)
    ]

    # run and print table
    print("="*60)
    print(f"{'Dataset':<10} | {'Attack Type':<18} | {'eps=0.05':<8} | {'eps=0.1':<8} | {'eps=0.2':<8} | {'eps=0.3':<8}")
    print("-"*60)

    for d_name, model, loader in datasets:
        for a_name, a_fn, is_t in attacks:
            row_str = f"{d_name:<10} | {a_name:<18} | "
            
            for eps in epsilons:
                # save img only when eps is 0.3
                save_vis = (eps == 0.3)
                
                sr = run_attack_eval(
                    model, loader, a_fn, is_targeted=is_t, 
                    eps=eps, device=device, 
                    dataset_name=d_name, attack_name=a_name.replace(" ", "_"), 
                    num_samples=100, save_vis=save_vis
                )
                row_str += f"{sr:>6.1f}% | "
                
            print(row_str[:-2])
            
    print("="*60)
