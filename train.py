"""
Universal training script for plant classification models
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
from datetime import datetime

# Import from our modules
from config import (
    BATCH_SIZE, NUM_EPOCHS, INITIAL_LR, NUM_CLASSES,
    DROPOUT_RATE, SAVE_DIR, DEVICE, PATIENCE,
    LABEL_SMOOTHING, WEIGHT_DECAY, MIXUP_ALPHA,
    CUTMIX_ALPHA, USE_MIXUP, USE_CUTMIX, TRAIN_DIR,
    VAL_DIR, NUM_WORKERS, PIN_MEMORY
)

from utils import (
    set_seed, create_data_loaders, train_epoch,
    val_epoch, count_parameters
)

from models import PlantClassifier, AdvancedPlantClassifier


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train plant classification model')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18',
                                 'resnet34', 'resnet50', 'resnext50'],
                        help='Backbone model architecture')
    parser.add_argument('--advanced', action='store_true',
                        help="""Use advanced
                        model architecture with GeM pooling""")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=INITIAL_LR,
                        help='Initial learning rate')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive resizing during training')
    parser.add_argument('--initial_size', type=int, default=200,
                        help='Initial image size for progressive resizing')
    parser.add_argument('--final_size', type=int, default=224,
                        help='Final image size')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    # Create directory for this model
    model_name = f"{args.model}_{'advanced' if args.advanced else 'basic'}"
    model_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Define image sizes based on arguments
    initial_size = args.initial_size if args.progressive else args.final_size
    final_size = args.final_size

    print(f"Training {model_name} model")
    print(f"""{'Using progressive resizing'
          if args.progressive else 'Using fixed image size'}""")

    # Phase 1: Initial image size
    print(f"Phase 1: Training with image size {initial_size}")

    # Define transforms for Phase 1
    train_transform_phase1 = transforms.Compose([
        transforms.RandomResizedCrop(initial_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=0,
                                translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform_phase1 = transforms.Compose([
        transforms.Resize(int(initial_size * 1.14)),
        transforms.CenterCrop(initial_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders for Phase 1
    train_loader_phase1, val_loader_phase1, class_to_idx = create_data_loaders(
        TRAIN_DIR, VAL_DIR, args.batch_size, NUM_WORKERS, PIN_MEMORY,
        train_transform_phase1, val_transform_phase1
    )

    # Phase 2 transforms (if using progressive resizing)
    if args.progressive:
        train_transform_phase2 = transforms.Compose([
            transforms.RandomResizedCrop(final_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=0,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform_phase2 = transforms.Compose([
            transforms.Resize(int(final_size * 1.14)),
            transforms.CenterCrop(final_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Initialize model
    if args.advanced:
        model = AdvancedPlantClassifier(args.model,
                                        num_classes=NUM_CLASSES,
                                        pretrained=True,
                                        dropout_rate=DROPOUT_RATE)
    else:
        model = PlantClassifier(args.model,
                                num_classes=NUM_CLASSES,
                                pretrained=True,
                                dropout_rate=DROPOUT_RATE)

    model = model.to(DEVICE)

    # Check parameter count
    model_params = count_parameters(model)
    print(f"Model parameters: {model_params:,}")
    print(f"Model is under the 100M limit: {model_params < 100e6}")

    # Loss function with mild label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler with warmup
    def warmup_cosine_schedule(epoch, warmup_epochs=5, max_epochs=30):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi *
                                     (epoch - warmup_epochs) /
                                     (max_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lambda epoch:
                                                warmup_cosine_schedule(epoch))

    # Training variables for Phase 1
    phase1_epochs = args.epochs // 2 if args.progressive else args.epochs
    best_val_acc = 0.0
    patience = PATIENCE
    no_improve_count = 0

    # Training history
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}

    # Phase 1 Training loop
    for epoch in range(phase1_epochs):
        # Gradually reduce mixup and cutmix alpha as training progresses
        mixup_alpha = max(0.1, MIXUP_ALPHA - epoch * 0.01)
        cutmix_alpha = max(0.5, CUTMIX_ALPHA - epoch * 0.01)

        # Adjust mixup/cutmix probabilities
        mixup_prob = 0.5 * (1 - epoch / phase1_epochs)
        cutmix_prob = 0.5 * (1 - epoch / phase1_epochs)

        train_loss, train_acc = train_epoch(
            model, train_loader_phase1, criterion, optimizer, DEVICE,
            use_mixup=USE_MIXUP, mixup_alpha=mixup_alpha,
            use_cutmix=USE_CUTMIX, cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob, cutmix_prob=cutmix_prob
        )
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader_phase1,
                                              criterion, DEVICE)

        # Update learning rate
        scheduler.step()

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Print epoch results
        print(f"Epoch {epoch+1}/{phase1_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"""Val Loss: {val_loss:.4f},
              Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}""")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            # Save best model
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best_model_phase1.pth'))
            print(f"Model saved with validation acc: {val_acc:.4f}")
        else:
            no_improve_count += 1

        # Check for early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Phase 2 (if using progressive resizing)
    if args.progressive:
        print(f"Phase 2: Training with image size {final_size}")

        # Create data loaders for Phase 2
        train_loader_phase2, val_loader_phase2, _ = create_data_loaders(
            TRAIN_DIR, VAL_DIR, args.batch_size, NUM_WORKERS, PIN_MEMORY,
            train_transform_phase2, val_transform_phase2
        )

        # Load best model from Phase 1
        path = os.path.join(model_dir, 'best_model_phase1.pth')
        model.load_state_dict(torch.load(path))

        # Reset optimizer and scheduler for Phase 2
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr/2, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6
        )

        # Reset training variables for Phase 2
        phase2_epochs = args.epochs - phase1_epochs
        best_val_acc = 0.0
        no_improve_count = 0

        # Phase 2 Training history
        phase2_history = {'train_loss': [], 'train_acc': [],
                          'val_loss': [], 'val_acc': [], 'val_f1': []}

        # Phase 2 Training loop
        for epoch in range(phase2_epochs):

            # Less augmentation in Phase 2
            mixup_alpha = max(0.05, 0.2 - epoch * 0.01)
            cutmix_alpha = max(0.2, 0.5 - epoch * 0.01)
            mixup_prob = 0.3
            cutmix_prob = 0.3

            train_loss, train_acc = train_epoch(
                model, train_loader_phase2, criterion, optimizer, DEVICE,
                use_mixup=USE_MIXUP, mixup_alpha=mixup_alpha,
                use_cutmix=USE_CUTMIX, cutmix_alpha=cutmix_alpha,
                mixup_prob=mixup_prob, cutmix_prob=cutmix_prob
            )
            val_loss, val_acc, val_f1 = val_epoch(model, val_loader_phase2,
                                                  criterion, DEVICE)

            # Update learning rate
            scheduler.step()

            # Update history
            phase2_history['train_loss'].append(train_loss)
            phase2_history['train_acc'].append(train_acc)
            phase2_history['val_loss'].append(val_loss)
            phase2_history['val_acc'].append(val_acc)
            phase2_history['val_f1'].append(val_f1)

            # Also update overall history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            # Print epoch results
            print(f"Phase 2 - Epoch {epoch+1}/{phase2_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"""Val Loss: {val_loss:.4f},
                  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}""")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
                # Save best model
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'best_model.pth'))
                print(f"Model saved with validation acc: {val_f1:.4f}")
            else:
                no_improve_count += 1

            # Check for early stopping
            if no_improve_count >= patience:
                print(f"Early stopping triggered at phase 2 epoch {epoch+1}")
                break
    else:
        # If not using progressive resizing, rename the best model
        os.rename(
            os.path.join(model_dir, 'best_model_phase1.pth'),
            os.path.join(model_dir, 'best_model.pth')
        )

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    if args.progressive:
        plt.axvline(x=phase1_epochs-1,
                    color='r', linestyle='--', label='Phase Change')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    if args.progressive:
        plt.axvline(x=phase1_epochs-1,
                    color='r', linestyle='--', label='Phase Change')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Validation F1 Score')
    if args.progressive:
        plt.axvline(x=phase1_epochs-1,
                    color='r', linestyle='--', label='Phase Change')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))

    # Load best model and evaluate
    print("Loading best model for evaluation")
    model.load_state_dict(torch.load(os.path.join(model_dir,
                                                  'best_model.pth')))

    final_loader = val_loader_phase2 if args.progressive else val_loader_phase1

    val_loss, val_acc, val_f1 = val_epoch(model,
                                          final_loader, criterion, DEVICE)
    print(f"""Final validation - Loss: {val_loss:.4f},
          Acc: {val_acc:.4f}, F1: {val_f1:.4f}""")

    # Save model metadata
    metadata = {
        'model_name': model_name,
        'backbone': args.model,
        'advanced': args.advanced,
        'parameters': model_params,
        'final_val_acc': val_acc,
        'final_val_f1': val_f1,
        'epochs_trained': len(history['train_loss']),
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    pd.DataFrame([metadata]).to_csv(os.path.join(model_dir,
                                                 'metadata.csv'), index=False)

    print(f"""Training completed. Model saved at {os.path.join(model_dir,
          'best_model.pth')}""")


if __name__ == '__main__':
    main()
