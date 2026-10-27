import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """Training loop from the module notebooks, with per-epoch and best-model checkpoints."""
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix(
                    {
                        "avg accuracy": f"{running_correct/running_total:.3f}",
                        "avg loss": f"{running_loss/(batch_number+1):.4f}",
                    }
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        # Validation metrics
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, checkpoint_dir
        )

        # Save best model (by validation accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best"
            )
            print(f"New best model saved! Val accuracy: {val_accuracy:.2f}%")

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_accuracy:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished Training")
    return model


def train_gan(model, data_loader, criterion=None, optimizer=None, device='cpu', epochs=10):
    """
    Standard GAN training loop (Module 6 class activity).

    model:     GAN from get_model("GAN") (has .generator and .discriminator)
    criterion: defaults to nn.BCELoss() (the discriminator outputs a probability)
    optimizer: (opt_generator, opt_discriminator); defaults to Adam(lr=2e-4, betas=(0.5, 0.999))
    """
    import torch.nn as nn

    gen, disc = model.generator.to(device), model.discriminator.to(device)
    criterion = criterion or nn.BCELoss()
    if optimizer is None:
        optimizer = (
            torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)),
        )
    opt_g, opt_d = optimizer
    history = []

    for epoch in range(epochs):
        gen.train(); disc.train()
        progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real, _) in enumerate(progress):
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # --- Train discriminator: real -> 1, fake -> 0 ---
            z = torch.randn(b, model.z_dim, device=device)
            fake = gen(z)
            loss_d = criterion(disc(real), ones) + criterion(disc(fake.detach()), zeros)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- Train generator: make the discriminator call fakes real ---
            loss_g = criterion(disc(fake), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            if batch_number % 100 == 0:
                progress.set_postfix({"D loss": f"{loss_d.item():.4f}", "G loss": f"{loss_g.item():.4f}"})
        history.append({"epoch": epoch + 1, "d_loss": loss_d.item(), "g_loss": loss_g.item()})
        print(f"Epoch {epoch+1}: D loss={loss_d.item():.4f}, G loss={loss_g.item():.4f}")

    model.history = history
    return model
