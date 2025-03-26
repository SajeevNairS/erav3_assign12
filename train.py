import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from train_get2_8_init import GPT, GPTConfig, DataLoaderLite
import time
from tqdm import tqdm

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Model configuration
config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,  # 124M parameters
    n_head=12,
    n_embd=768
)

# Training hyperparameters
batch_size = 32
sequence_length = 64
learning_rate = 1e-4
warmup_steps = 1000
max_steps = 10000
eval_interval = 100
save_interval = 1000

# Initialize model and move to device
model = GPT(config)
model.to(device)

# Initialize optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.1
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=1,
    epochs=max_steps,
    pct_start=0.1
)

# Data loader
train_loader = DataLoaderLite(B=batch_size, T=sequence_length)

# Training loop
best_loss = float('inf')
for step in tqdm(range(max_steps)):
    # Get batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()

    # Logging
    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}, lr = {scheduler.get_last_lr()[0]:.2e}")

    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'best_model.pt')
        print(f"New best loss: {best_loss:.4f}")

    # Early stopping if loss is below target
    if loss.item() < 0.099999:
        print(f"Target loss achieved at step {step}: {loss.item():.4f}")
        break

print(f"Training completed. Best loss: {best_loss:.4f}") 