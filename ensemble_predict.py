"""
Enhanced ensemble prediction script for plant classification models
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from sklearn.metrics import f1_score

# Import from our modules
from config import (
    SAVE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, NUM_CLASSES, NUM_TTA, DEVICE
)
from utils import (
    set_seed, get_class_mapping, create_data_loaders, tta_predict
)
from models import PlantClassifier, AdvancedPlantClassifier


def evaluate_model_performance(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    softmax_outputs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            softmax_outputs.append(probs.cpu())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Get calibration score (proxy for model confidence calibration)
    all_softmax = torch.cat(softmax_outputs, dim=0).numpy()
    confidence = np.max(all_softmax, axis=1)
    calibration_gap = np.mean(confidence) - accuracy

    return accuracy, f1, calibration_gap, np.mean(confidence)


def calibrate_predictions(probs, temperature=1.0):
    """Apply temperature scaling to calibrate model predictions"""
    if temperature == 1.0:
        return probs

    # Apply temperature scaling
    logits = torch.log(probs / (1 - probs + 1e-7))
    calibrated_logits = logits / temperature
    calibrated_probs = torch.sigmoid(calibrated_logits)
    return calibrated_probs


def get_optimal_ensemble_weights(models, val_loader, device, method='dynamic'):
    if method == 'f1_score':
        # Weight based on F1 scores
        model_scores = []
        for model in models:
            accuracy, f1, _, _ = evaluate_model_performance(model,
                                                            val_loader, device)
            model_scores.append(f1)

        # Calculate weights based on F1 scores
        total_score = sum(model_scores)
        weights = [score / total_score for score in model_scores]

    elif method == 'dynamic':
        # Get validation predictions from each model
        model_preds = []
        true_labels = []

        for i, model in enumerate(models):
            model.eval()
            batch_preds = []

            if i == 0:  # Only collect labels once
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        batch_preds.append(probs.cpu())
                        true_labels.extend(labels.numpy())

            else:
                with torch.no_grad():
                    for inputs, _ in val_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        batch_preds.append(probs.cpu())

            model_preds.append(torch.cat(batch_preds, dim=0))

        # Optimize weights using a simple grid search
        best_weights = None
        best_f1 = 0

        # Generate combinations of weights
        weight_options = np.linspace(0, 1, 6)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if len(models) <= 3:
            from itertools import product
            weight_combinations = list(product(weight_options,
                                               repeat=len(models)))
        else:  # For more models, use a more efficient approach
            # Generate 100 random weight combinations
            weight_combinations = []
            for _ in range(100):
                weights = np.random.dirichlet(np.ones(len(models)))
                weight_combinations.append(weights)

        for weights in weight_combinations:
            # Normalize weights to sum to 1
            norm_weights = np.array(weights) / sum(weights)
            if sum(norm_weights) == 0:
                continue

            # Apply weighted ensemble
            weighted_preds = torch.zeros_like(model_preds[0])
            for model_prob, weight in zip(model_preds, norm_weights):
                weighted_preds += model_prob * weight

            # Get predictions
            _, final_preds = torch.max(weighted_preds, dim=1)

            # Calculate F1 score
            current_f1 = f1_score(
                true_labels, final_preds.numpy(), average='weighted'
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_weights = norm_weights

        weights = best_weights
        print(f"Optimized ensemble F1 score: {best_f1:.4f}")

    else:  # Equal weighting
        weights = [1/len(models)] * len(models)

    return weights


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Ensemble prediction for plant classification'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['resnet18', 'resnet34', 'resnet50', 'resnext50'],
        help='List of model architectures to use'
    )
    parser.add_argument(
        '--advanced',
        nargs='+',
        default=[False, False, False, True],
        type=bool,
        help='Whether each model uses the advanced architecture'
    )
    parser.add_argument(
        '--weighting',
        choices=['f1_score', 'dynamic', 'equal'],
        default='dynamic',
        help='Method to determine model weights'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        default=True,
        help='Whether to apply temperature scaling for calibration'
    )
    parser.add_argument(
        '--diversity_filter',
        action='store_true',
        default=True,
        help='Apply diversity filtering to select complementary models'
    )
    parser.add_argument(
        '--tta',
        type=int,
        default=NUM_TTA,
        help='Number of test-time augmentations'
    )
    args = parser.parse_args()

    # Validate arguments
    if len(args.models) != len(args.advanced):
        raise ValueError("Number of models and advanced flags must match")

    # Set seed for reproducibility
    set_seed(42)

    # Load models
    models = []
    model_paths = []

    for i, (model_arch, is_advanced) in enumerate(zip(args.models,
                                                      args.advanced)):
        # Construct model name
        model_name = f"{model_arch}_{'advanced' if is_advanced else 'basic'}"
        model_dir = os.path.join(SAVE_DIR, model_name)
        model_path = os.path.join(model_dir, 'best_model.pth')

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_name} not found at {model_path}")
            continue

        # Initialize model
        print(f"Loading model: {model_name}")
        if is_advanced:
            model = AdvancedPlantClassifier(
                model_arch, num_classes=NUM_CLASSES, pretrained=True
            )
        else:
            model = PlantClassifier(
                model_arch, num_classes=NUM_CLASSES, pretrained=True
            )

        # Load weights
        model.load_state_dict(torch.load(model_path))
        model = model.to(DEVICE)
        model.eval()

        models.append(model)
        model_paths.append(model_path)

    if not models:
        raise ValueError("No valid models found. Please train models first.")

    # Load class mapping
    class_to_idx = get_class_mapping(TRAIN_DIR)

    # Create validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create validation loader for weight optimization
    _, val_loader, _ = create_data_loaders(
        TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
        val_transform, val_transform
    )

    # Apply diversity filtering if requested
    if args.diversity_filter and len(models) > 2:
        # Calculate pairwise disagreement matrix
        disagreement_matrix = np.zeros((len(models), len(models)))

        # Get predictions for all models on validation set
        model_val_preds = []
        for model in models:
            model.eval()
            all_preds = []

            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())

            model_val_preds.append(np.array(all_preds))

        # Calculate disagreement between model pairs
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    disagreement = np.mean(model_val_preds[i] !=
                                           model_val_preds[j])
                    disagreement_matrix[i, j] = disagreement

        model_diversity_scores = np.sum(disagreement_matrix, axis=1)

        # Get individual performances
        model_performances = []
        for model in models:
            _, f1, _, _ = evaluate_model_performance(model, val_loader, DEVICE)
            model_performances.append(f1)

        # Normalize scores
        norm_diversity = model_diversity_scores / np.max(
            model_diversity_scores)
        norm_performance = np.array(model_performances) / np.max(
            model_performances)

        # Combine diversity and performance (equal weighting)
        combined_score = 0.7 * norm_performance + 0.3 * norm_diversity

        # Select top models
        num_to_keep = max(2, len(models) - 1)  # Keep at least 2 or all but 1
        top_indices = np.argsort(combined_score)[-num_to_keep:]

        # Filter models
        filtered_models = [models[i] for i in top_indices]
        filtered_paths = [model_paths[i] for i in top_indices]

        # Use filtered models
        models = filtered_models
        model_paths = filtered_paths

        print(f"Selected {len(models)} diverse models after filtering")

    # Determine model weights
    model_weights = get_optimal_ensemble_weights(
        models, val_loader, DEVICE, method=args.weighting
    )
    print(f"Using model weights: {[round(w, 3) for w in model_weights]}")

    # Get model calibration temperatures if calibration is enabled
    if args.calibrate:
        print("Calibrating model predictions...")
        temperatures = []

        for model in models:
            _, _, calibration_gap, _ = evaluate_model_performance(
                model, val_loader, DEVICE
            )

            # Set temperature based on calibration gap
            if calibration_gap > 0.05:  # Overconfident model
                temp = 1.2  # Increase temperature to reduce confidence
            elif calibration_gap < -0.05:  # Underconfident model
                temp = 0.8  # Decrease temperature to increase confidence
            else:
                temp = 1.0  # Well-calibrated model

            temperatures.append(temp)
            print(f"Model calibration temperature: {temp:.2f}")
    else:
        temperatures = [1.0] * len(models)

    ensemble_predictions = []
    ensemble_ids = None

    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        _, ids, pred_probs = tta_predict(
            model, TEST_DIR, DEVICE, class_to_idx, num_augmentations=args.tta
        )

        # Apply temperature scaling for calibration
        calibrated_probs = calibrate_predictions(pred_probs, temperatures[i])

        # Save individual model predictions for reference
        model_name = os.path.basename(os.path.dirname(model_paths[i]))
        _, preds = torch.max(calibrated_probs, dim=1)

        # Convert to original class labels
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        pred_labels = [int(idx_to_class[pred.item()]) for pred in preds]

        pd.DataFrame({
            'image_name': ids,
            'pred_label': pred_labels
        }).to_csv(f'submission_{model_name}.csv', index=False)

        if ensemble_ids is None:
            ensemble_ids = ids

        # Store prediction probabilities for weighted ensemble
        ensemble_predictions.append((calibrated_probs, model_weights[i]))

    # Apply weighted ensemble
    weighted_predictions = torch.zeros_like(ensemble_predictions[0][0])
    for preds, weight in ensemble_predictions:
        weighted_predictions += preds * weight

    # Get final class predictions
    _, final_preds = torch.max(weighted_predictions, dim=1)

    # Convert to original class labels
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    final_pred_labels = [int(idx_to_class[pred.item()])
                         for pred in final_preds]

    # Create submission file
    submission = pd.DataFrame({
        'image_name': ensemble_ids,
        'pred_label': final_pred_labels
    })

    ensemble_name = '_'.join([m.split('_')[0] for m in args.models])
    submission_path = f'submission_ensemble_{ensemble_name}_improved.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Ensemble submission file saved as {submission_path}")


if __name__ == '__main__':
    main()
