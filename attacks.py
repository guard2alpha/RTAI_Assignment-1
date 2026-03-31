import torch

# 1. FGSM Targeted (타겟 오답으로 유도)
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    
    logits = model(x_adv)
    loss = torch.nn.functional.cross_entropy(logits, target)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        # minimize loss to target (minus sign)
        x_adv = x_adv - eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    return x_adv

# 2. FGSM Untargeted (정답에서 멀어지게 함)
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    
    logits = model(x_adv)
    loss = torch.nn.functional.cross_entropy(logits, label)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        # maximize loss from true label (plus sign)
        x_adv = x_adv + eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    return x_adv

# 3. PGD Targeted (여러 번 쪼개서 타겟으로 유도)
def pgd_targeted(model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach()
    
    for _ in range(k):
        x_adv.requires_grad_(True)
        
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, target)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # step towards target (minus)
            x_adv = x_adv - eps_step * x_adv.grad.sign()
            
            # project to eps ball
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            
            # clamp to image range
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    return x_adv

# 4. PGD Untargeted (여러 번 쪼개서 정답에서 멀어지게 함)
def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach()
    
    for _ in range(k):
        x_adv.requires_grad_(True)
        
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, label)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # step away from true label (plus)
            x_adv = x_adv + eps_step * x_adv.grad.sign()
            
            # project to eps ball
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            
            # clamp to image range
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    return x_adv