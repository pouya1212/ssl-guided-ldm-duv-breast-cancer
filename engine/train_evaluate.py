"""
Module for training, validating, and testing a patch-based 
classification model for Whole Slide Images (WSI).
"""

import os
import time
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from utils.metrics import AverageMeter, simple_accuracy
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dist_util import get_world_size
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(args, model, fold):
    """Save model checkpoint."""
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint_path = os.path.join(
        args.output_dir, f"{args.name}_fold{fold}_checkpoint_with_weight.bin"
    )
    torch.save(model_to_save.state_dict(), checkpoint_path)
    logger.info("Saved model checkpoint to [ %s ]", checkpoint_path)

def test(model, test_loader):
    """
    Evaluates the model on the provided test loader.
    """
    model.eval()
    correct = 0
    total = 0

    predictions = []
    true_labels = []
    all_alternative_labels = []
    top_probabilities = []
    patches_names = []
    wsi_names = []
    patches_indices = []
    patches_coordinates = []

    with torch.no_grad():
        for images, (labels, alt_labels), img_names, meta_info in tqdm(
            test_loader, desc="Testing", leave=False
        ):
            images = images.to('cuda')
            labels = labels.to('cuda')
            alt_labels = alt_labels.to('cuda')

            outputs = model(images)[0]
            probs = F.softmax(outputs, dim=-1)
            top_probs, predicts = torch.max(probs, dim=-1)

            total += labels.size(0)
            correct += (predicts == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            all_alternative_labels.extend(alt_labels.cpu().numpy())
            top_probabilities.extend(top_probs.cpu().numpy())
            predictions.extend(predicts.cpu().numpy())

            patches_names.extend(list(img_names))
            wsi_names.extend(list(meta_info[0]))
            patches_indices.extend(list(meta_info[1]))

            coordinates = meta_info[2]
            for i in range(len(coordinates[0])):
                x_coord = coordinates[0][i].item()
                y_coord = coordinates[1][i].item()
                patches_coordinates.append((x_coord, y_coord))

    accuracy = 100 * correct / total
    return (accuracy, patches_names, true_labels, all_alternative_labels,
            predictions, top_probabilities, wsi_names, patches_indices,
            patches_coordinates)

def valid(args, model, writer, val_loader, global_step):
    """
    Validates the model performance during training.
    """
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(val_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()

    for _, batch in enumerate(epoch_iterator):
        images, (labels, _), _, (_, _, _) = batch
        images = images.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            logits = model(images)[0]
            eval_loss = loss_fct(logits, labels)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], labels.detach().cpu().numpy(), axis=0)

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:2.5f})")

    accuracy = simple_accuracy(all_preds[0], all_label[0])

    logger.info("\nValidation Results")
    logger.info("Global Steps: %d", global_step)
    logger.info("Validation Loss: %2.5f", eval_losses.avg)
    logger.info("Validation Accuracy: %2.5f", accuracy)

    writer.add_scalar("validation/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg

def train_test(args, model, train_loader, val_loader, test_loader, tune_loader, fold):
    """
    Core function to handle Training, Validation, and Testing cycles.
    """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_steps_per_epoch = len(train_loader)
    t_total = train_steps_per_epoch * args.num_epochs

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        # Accessing protected members as required by some apex versions
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    start_time = time.time()
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    set_seed(args)

    losses = AverageMeter()
    train_losses, val_losses, val_accuracies, train_epoch_losses = [], [], [], []
    global_step, best_acc = 0, 0

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            images, (labels, _), _, (_, _, _) = batch
            images = images.to(args.device)
            labels = labels.to(args.device)

            _, _, loss = model(images, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                   args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                train_losses.append(losses.val)

                epoch_iterator.set_description(
                    f"Training ({global_step} / {t_total} Steps) (loss={losses.val:2.5f})"
                )

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)

                if global_step % train_steps_per_epoch == 0:
                    train_epoch_losses.append(losses.avg)
                    if args.local_rank in [-1, 0]:
                        acc, v_loss = valid(args, model, writer, val_loader, global_step)
                        val_losses.append(v_loss)
                        val_accuracies.append(acc)
                        if best_acc < acc:
                            save_model(args, model, fold)
                            best_acc = acc
                        model.train()

            if global_step % t_total == 0:
                break
        losses.reset()
        if global_step % t_total == 0:
            break

    # Load Best Model
    if args.local_rank in [-1, 0]:
        checkpoint_dir = os.path.join(
            args.output_dir, f"{args.name}_fold{fold}_checkpoint_with_weight.bin"
        )
        model.load_state_dict(torch.load(checkpoint_dir))
        writer.close()

    # Evaluation
    test_res = test(model, test_loader)
    tune_res = test(model, tune_loader)

    total_duration = time.time() - start_time
    logger.info("Total duration: %f seconds", total_duration)

    # Threshold Tuning
    patch_df = pd.DataFrame({
        "Patch_Name": tune_res[1],
        "True_patch_Label": tune_res[2],
        "Prediction": tune_res[4],
        "Top_probability": tune_res[5],
        "WSI_Name": tune_res[6],
        "Patch_Index": tune_res[7],
        "Coordinate": tune_res[8]
    })

    tune_dir = os.path.join(args.output_dir, "tune_predictions")
    os.makedirs(tune_dir, exist_ok=True)
    patch_df.to_csv(os.path.join(tune_dir, f"fold{fold}_patch_level_tune_predictions.csv"),
                    index=False)

    # Search for best thresholds
    best_majority = (0.0, 0.0)
    best_softmax = (0.0, 0.0)
    ths = np.arange(0.05, 0.64, 0.01)
    ths2 = np.arange(0.05, 0.27, 0.01)

    for t in ths:
        y_true, y_pred = [], []
        for _, grp in patch_df.groupby("WSI_Name"):
            y_true.append(int(grp["True_patch_Label"].any()))
            y_pred.append(int(grp["Prediction"].mean() >= t))
        acc = float((np.array(y_true) == np.array(y_pred)).mean())
        if acc >= best_majority[0]:
            best_majority = (acc, t)

    for t in ths2:
        y_true, y_pred = [], []
        for _, grp in patch_df.groupby("WSI_Name"):
            y_true.append(int(grp["True_patch_Label"].any()))
            weights = grp["Top_probability"].to_numpy()
            fusion = np.where(grp["Prediction"] == 1, 1, -1) * weights
            y_pred.append(int(fusion.mean() > t))
        acc = float((np.array(y_true) == np.array(y_pred)).mean())
        if acc >= best_softmax[0]:
            best_softmax = (acc, t)

    fold_thresholds = {"majority": best_majority[1], "softmax": best_softmax[1]}
    with open(os.path.join(tune_dir, f"fold{fold}_thresholds.json"), "w") as jf:
        json.dump(fold_thresholds, jf)

    return (train_epoch_losses, train_losses, val_losses, val_accuracies,
            best_acc, test_res[0], test_res[1], test_res[2], test_res[3],
            test_res[4], test_res[5], test_res[6], test_res[7], test_res[8],
            fold_thresholds)
