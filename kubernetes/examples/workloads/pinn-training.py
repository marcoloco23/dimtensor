#!/usr/bin/env python3
"""
PINN Training Workload for Kubernetes

This script trains a Physics-Informed Neural Network (PINN) to solve
the 1D heat equation: u_t = alpha * u_xx

The PINN learns to satisfy both the PDE and boundary/initial conditions
using dimensional analysis with dimtensor.

Usage:
    python pinn-training.py --epochs 1000 --checkpoint-dir /checkpoints
"""

import argparse
import os
import torch
import torch.nn as nn
from dimtensor.torch import DimTensor
from dimtensor import units


class HeatEquationPINN(nn.Module):
    """
    Physics-Informed Neural Network for 1D heat equation.

    Inputs: (x, t) - position and time
    Output: u(x, t) - temperature field
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t):
        """Forward pass: compute u(x, t)"""
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def compute_pde_loss(model, x, t, alpha, device):
    """
    Compute physics loss: residual of heat equation u_t = alpha * u_xx

    Args:
        model: PINN model
        x: spatial positions (with units)
        t: time points (with units)
        alpha: thermal diffusivity (with units)
        device: torch device
    """
    # Enable gradient computation
    x_data = x.data.requires_grad_(True)
    t_data = t.data.requires_grad_(True)

    # Forward pass
    u = model(x_data, t_data)

    # Compute derivatives using autograd
    u_t = torch.autograd.grad(u, t_data, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_data, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_data, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    # PDE residual: u_t - alpha * u_xx = 0
    # Dimensional analysis ensures correct units
    pde_residual = u_t - alpha.data[0] * u_xx

    return (pde_residual ** 2).mean()


def compute_ic_loss(model, x, device):
    """Initial condition loss: u(x, 0) = sin(pi * x)"""
    t_zero = torch.zeros_like(x.data)
    u_pred = model(x.data, t_zero)
    u_true = torch.sin(torch.pi * x.data)
    return ((u_pred - u_true) ** 2).mean()


def compute_bc_loss(model, t, device):
    """Boundary condition loss: u(0, t) = u(1, t) = 0"""
    x_left = torch.zeros_like(t.data)
    x_right = torch.ones_like(t.data)

    u_left = model(x_left, t.data)
    u_right = model(x_right, t.data)

    return (u_left ** 2).mean() + (u_right ** 2).mean()


def train_pinn(epochs=1000, batch_size=256, lr=0.001, checkpoint_dir="/checkpoints"):
    """Train PINN for heat equation"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    model = HeatEquationPINN(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Thermal diffusivity (with units)
    alpha = DimArray([0.01], units.m**2 / units.s)

    print(f"Starting training for {epochs} epochs...")
    print(f"Thermal diffusivity: {alpha}")

    best_loss = float('inf')

    for epoch in range(epochs):
        # Sample random points in domain [0, 1] x [0, 1]
        x = DimTensor(torch.rand(batch_size, 1, device=device), units.m, requires_grad=True)
        t = DimTensor(torch.rand(batch_size, 1, device=device), units.s, requires_grad=True)

        # Compute losses
        pde_loss = compute_pde_loss(model, x, t, alpha, device)
        ic_loss = compute_ic_loss(model, x, device)
        bc_loss = compute_bc_loss(model, t, device)

        # Total loss (weighted combination)
        total_loss = pde_loss + 10 * ic_loss + 10 * bc_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  PDE loss: {pde_loss.item():.6f}")
            print(f"  IC loss:  {ic_loss.item():.6f}")
            print(f"  BC loss:  {bc_loss.item():.6f}")
            print(f"  Total:    {total_loss.item():.6f}")

        # Save best checkpoint
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_path)

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
    }, final_path)

    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train PINN for heat equation")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="/checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    train_pinn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
