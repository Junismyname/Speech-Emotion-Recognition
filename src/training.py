import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def default_forward(model, inputs):
    return model(inputs)

def train_model(model, train_loader, val_loader, optimizer, device, epochs, model_name="Model",
                scheduler=None, patience=None, forward=default_forward, augment=None, criterion=None,
                batch_scheduler=None, max_grad_norm=None):
    """Shared training loop for every PyTorch model.

    Model selection uses the validation split only: the weights with the best val accuracy
    are restored at the end, and early stopping (if `patience` is set) watches val accuracy.
    The test split is never seen here.

    `augment` (optional) is applied to training batches only, on the device.
    `batch_scheduler` (e.g. warmup + cosine) steps after every batch instead of every epoch, and
    `max_grad_norm` clips gradients; both keep fine-tuning of large pre-trained models stable.
    `criterion` defaults to plain cross-entropy; validation loss always uses plain cross-entropy
    so it stays comparable across models.
    """
    print(f"\n--- Training {model_name} ---")
    val_criterion = nn.CrossEntropyLoss()
    criterion = criterion or val_criterion
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if augment is not None:
                inputs = augment(inputs)
            optimizer.zero_grad()
            outputs = forward(model, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if batch_scheduler is not None:
                batch_scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = forward(model, inputs)
                val_loss += val_criterion(outputs, labels).item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        for key, value in zip(history, [epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc]):
            history[key].append(value)

        # ReduceLROnPlateau watches val loss; other schedulers step once per epoch
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        elif scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} | val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Kept in CPU RAM so large models (ViT, WavLM) don't hold a second copy of their weights in VRAM
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                break

    model.load_state_dict(best_weights)
    print(f"Restored best weights with Val Accuracy: {best_val_acc * 100:.2f}%")
    return history

def predict(model, loader, device, forward=default_forward):
    """Returns (predicted indices, softmax probabilities) for every batch in the loader."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = forward(model, inputs.to(device))
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs)
    return probs.argmax(axis=1), probs
