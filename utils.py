"""
Utility functions for plant classification models
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Mixup augmentation implementation
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# CutMix augmentation implementation
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Get bbox for cutmix
    W, H = x.size()[2], x.size()[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Copy and paste
    x_1 = x.clone()
    x_1[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return x_1, y, y[index], lam


# Custom dataset for test data
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = torchvision.io.read_image(img_name)
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        img_id = self.image_files[idx].split('.')[0]
        return image, img_id


# Create class-to-idx mapping that preserves numerical order
def get_class_mapping(train_dir):
    sorted_classes = sorted(os.listdir(train_dir), key=int)
    class_to_idx = {cls_name: idx for idx, cls_name in
                    enumerate(sorted_classes)}
    return class_to_idx


# Count class distribution
def get_class_counts(train_dir):
    classes = os.listdir(train_dir)
    class_counts = {}
    for c in classes:
        class_counts[c] = len(os.listdir(os.path.join(train_dir, c)))
    return class_counts


# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Create data loaders with proper class balancing
def create_data_loaders(train_dir, val_dir, batch_size,
                        num_workers=4, pin_memory=True,
                        train_transform=None, val_transform=None):
    # Get class mapping and counts
    class_to_idx = get_class_mapping(train_dir)
    class_counts = get_class_counts(train_dir)

    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )
    train_dataset.class_to_idx = class_to_idx
    train_dataset.samples = [(sample[0],
                              class_to_idx[train_dataset.classes[sample[1]]])
                             for sample in train_dataset.samples]

    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )
    val_dataset.class_to_idx = class_to_idx
    val_dataset.samples = [(sample[0],
                            class_to_idx[val_dataset.classes[sample[1]]])
                           for sample in val_dataset.samples]

    # Create weighted sampler for imbalanced classes
    class_weights = {c: 1.0/count for c, count in class_counts.items()}
    sample_weights = [class_weights[train_dataset.classes[label]] for _,
                      label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, class_to_idx


# Training function with mixup and cutmix
def train_epoch(model, data_loader, criterion, optimizer, device,
                use_mixup=True, mixup_alpha=0.2,
                use_cutmix=True, cutmix_alpha=1.0,
                mixup_prob=0.5, cutmix_prob=0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Randomly apply either mixup or cutmix
        r = np.random.rand()
        if r < mixup_prob and use_mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs,
                                                         labels, mixup_alpha)
            use_mix = 'mixup'
        elif r < mixup_prob + cutmix_prob and use_cutmix:
            inputs, labels_a, labels_b, lam = cutmix_data(inputs,
                                                          labels, cutmix_alpha)
            use_mix = 'cutmix'
        else:
            labels_a, labels_b = labels, labels
            lam = 1.0
            use_mix = None

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mix:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # For accuracy calculation, we'll use the primary labels
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        else:
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Validation function
def val_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Calculate and return F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, f1


# Test Time Augmentation (TTA) implementation
def tta_predict(model, test_dir, device, class_to_idx, num_augmentations=10):
    # Define TTA transforms
    tta_transforms = [
        # Original center crop
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0,
                                   saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Contrast adjustment
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0,
                                   contrast=0.2, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Saturation adjustment
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0,
                                   contrast=0, saturation=0.2, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Larger resize + center crop
        transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Smaller resize + center crop
        transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Random crop
        transforms.Compose([
            transforms.Resize(280),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Rotation
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    ]

    test_dataset = TestDataset(
        test_dir,
        transform=tta_transforms[0]  # Start with the default transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model.eval()

    # Store image IDs (need to get them only once)
    image_ids = []
    for _, ids in test_loader:
        image_ids.extend(ids)

    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Apply each transformation and accumulate predictions
    all_predictions = []

    iterate = enumerate(tta_transforms[:num_augmentations])
    for transform_idx, transform in iterate:
        print(f"""Applying TTA transform
              {transform_idx+1}/{len(tta_transforms[:num_augmentations])}""")

        # Update the transform
        test_dataset.transform = transform

        # Get predictions for this transform
        predictions_for_transform = []

        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, desc=f"TTA {transform_idx+1}"):
                inputs = inputs.to(device)

                # Get predictions
                outputs = model(inputs)

                # Store raw probabilities
                predictions_for_transform.append(F.softmax(outputs,
                                                           dim=1).cpu())

        # Concat batch results
        transform_predictions = torch.cat(predictions_for_transform, dim=0)
        all_predictions.append(transform_predictions)

    # Average predictions across all transforms
    avg_predictions = torch.stack(all_predictions).mean(dim=0)

    # Get final class predictions
    _, final_preds = torch.max(avg_predictions, dim=1)

    # Convert to original class labels
    final_pred_labels = [int(idx_to_class[pred.item()])
                         for pred in final_preds]

    return final_pred_labels, image_ids, avg_predictions
