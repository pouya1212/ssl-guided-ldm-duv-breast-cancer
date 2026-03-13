import os                 # File and directory management
import time               # Timing for performance measurement
import json               # Save/load thresholds as JSON
import logging            # Logging training/validation info
import itertools          # Itertools for loops or combinations
import torch
import torch.nn.functional as F              # Softmax, activation functions, etc.
from torch.utils.data import DataLoader      # Handling datasets and batches
from torch.utils.tensorboard import SummaryWriter  # Logging metrics
from apex import amp                          # Mixed precision training
from apex.parallel import DistributedDataParallel as DDP  # Multi-GPU distributed training
import torch.distributed as dist              # Distributed process utilities
import numpy as np                             # Numerical operations
from tqdm import tqdm                          # Progress bars
import pandas as pd                            # DataFrames for saving patch-level predictions
from utils.metrics import AverageMeter, simple_accuracy  # Custom metrics
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule  # Learning rate schedulers
from utils.dist_util import get_world_size     # Helper for distributed training

#pip install git+https://github.com/NVIDIA/apex.git

def test(model, test_loader):
    model.eval()  # Set model to evaluation mode
    
    correct = 0
    total = 0
    
    predictions = []  # List storing the predictions of the batches of images
    true_labels = []  # List storing the true_labels of the batches of images (0 for benign and 1 for malignant)
    all_Altertnative_labels = []  # List storing the alternative true_labels of the batches of images
    top_probabilities = []  # List storing top probabilities (for confidence)
    patches_names = []  # List storing the image_names of the batches
    wsi_names = []  # List storing the wsi names of the patches in the batches
    patches_indices = []  # List storing the indices of the patches names in the batches of images
    patches_coordinates = []  # List to store all coordinates across all batches
    # all_patch_densenet_gradcam_importance_weights = [] # list to store all densenet gradcam importance weights for patches
    with torch.no_grad():
        for images, (labels, Alt_labels), img_names, meta_info in tqdm(test_loader, desc="Testing", leave=False):
            images, labels, Alt_labels = images.to('cuda'), labels.to('cuda'), Alt_labels.to('cuda')
            
            # Forward pass
            outputs = model(images)[0]
            
            # Compute softmax probabilities
            probs = F.softmax(outputs, dim=-1)  # Apply softmax to get probabilities
            # Find the highest probability and its corresponding class
            top_probs, predicts = torch.max(probs, dim=-1)  # top_prob is the confidence, top_class is the predicted label

            # Update counts for accuracy
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
            
            # Collect results
            true_labels.extend(labels.cpu().numpy())  # true labels
            all_Altertnative_labels.extend(Alt_labels.cpu().numpy())
            top_probabilities.extend(top_probs.cpu().numpy())  # max probability per sample
            predictions.extend(predicts.cpu().numpy())  # predicted class labels
            patches_names.extend([name for name in img_names])
            wsi_names.extend([sample_name for sample_name in meta_info[0]])
            patches_indices.extend([index for index in meta_info[1]])
            
            # Store coordinates for each image in the batch
            coordinates = meta_info[2]  # Get coordinates for the batch
            for i in range(len(coordinates[0])):  # Iterate over each image in the batch
                x = coordinates[0][i].item()  # Convert tensor to a number
                y = coordinates[1][i].item()  # Convert tensor to a number
                patches_coordinates.append((x, y))  # Store (x, y) tuple for the image

    # Calculate accuracy
    accuracy = 100 * correct / total
    # print(f'Test Accuracy: {accuracy:.2f}%')

    # Return the results
    return accuracy, patches_names, true_labels, all_Altertnative_labels, predictions, top_probabilities, wsi_names, patches_indices, patches_coordinates

###############################################################################################################################       
# This function validate the model performance on the validation set during the training

def valid(args, model, writer, val_loader, global_step):
    # Validation!
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
    
    for step, batch in enumerate(epoch_iterator): # STEP IS INDEX OF BATCH, 
        # Unpack the batch and move only tensors to the device
        
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
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Validation Loss: %2.5f" % eval_losses.avg)
    logger.info("Validation Accuracy: %2.5f" % accuracy)

    writer.add_scalar("validation/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy , eval_losses.avg

################################################################################################################################################################
# This function applyig, Training, validation and testing

def train_test(args, model, train_loader, val_loader, test_loader, tune_loader, fold):
    """ Train the model """
    if args.local_rank in [-1, 0]: #This line checks the value of args.local_rank to determine whether the current process is either the main process or a non-distributed process.
        os.makedirs(args.output_dir, exist_ok=True) #-1 usually indicates non-distributed training, meaning only a single GPU is being used. #0 refers to the main process when distributed training is enabled (multi-GPU training). Only the main process handles certain tasks, such as logging, saving models, etc
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps #args.gradient_accumulation_steps: This is the number of forward and backward passes to accumulate gradients before performing a weight update. When memory is limited, using gradient accumulation allows you to simulate a larger effective batch size by updating the model after accumulating gradients over multiple mini-batches.
                                                                                       #The batch size per step would be 32 // 4 = 8. So, the model will process 8 samples per step, but it will accumulate the gradients from 4 such steps,
    train_steps_per_epoch = len(train_loader) # len(train_loader) = 50: This means the training dataset has been divided into 50 batches, each of size 512 (except possibly the last batch, if the total number of samples isn't divisible by 512).

    t_total = train_steps_per_epoch * args.num_epochs
                                                                                      # #and only then will it perform a weight update, simulating an effective batch size of 32.

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) #warmup_steps: Specifies the number of initial steps during which the learning rate linearly increases from a small value to the full learning rate, followed by decay.
    else: #Warmup helps avoid large, unstable updates early in training, especially in scenarios like large-batch training or when using adaptive optimizers.
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16: #This checks if mixed precision training is enabled (i.e., fp16 stands for "floating-point 16-bit precision").
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level) #Initializes the model and optimizer to support mixed precision training.
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20 # This manually sets the initial loss scale for mixed precision training.
    # Distributed training
    #This block checks if distributed training is enabled by inspecting args.local_rank, which represents the rank (ID) of the current process in a multi-process setup. If args.local_rank is not -1, the model will be wrapped in PyTorch's Distributed Data Parallel (DDP) module for distributed training.
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    
    # Capture start time
    start_time = time.time()  # Start timing    

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    
    losses = AverageMeter()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_epoch_losses = []
    epoch_average_losses = []
    
    global_step, best_acc = 0, 0 
    
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            images, (labels, _), _, (_, _, _) = batch


            # Move only the image and label tensors to the device
            images = images.to(args.device)
            labels = labels.to(args.device)
            # Pass images and labels to the model
            _, _, loss = model(images, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                #store train loss
                train_losses.append(losses.val)

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]: #This checks if the script is running in a non-distributed mode (i.e., args.local_rank == -1) or if it is the master process in a distributed setting (where args.local_rank == 0).
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    
                # Check if global_step is a multiple of 50 to extract the average
                if global_step % train_steps_per_epoch == 0:  # Change 50 to train_steps_per_epoch if needed
                    avg_loss_last_steps = losses.avg  # This gives you the average of the last 50 steps
                    train_epoch_losses.append(avg_loss_last_steps)
                    print(f"Average Loss for Steps {global_step - train_steps_per_epoch} to {global_step}: {avg_loss_last_steps:.4f}")
                
                    
                if global_step % train_steps_per_epoch == 0 and args.local_rank in [-1, 0]:
                    accuracy, val_loss = valid(args, model, writer, val_loader, global_step)
                    print(f" Epoch {global_step // train_steps_per_epoch}, Train Loss: {loss}, Val Loss: {val_loss}, Val Accuracy: {accuracy}")
                    val_losses.append(val_loss)
                    val_accuracies.append(accuracy)
                    
                    if best_acc < accuracy:
                        save_model(args, model, fold)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
                
        losses.reset()
        if global_step % t_total == 0:
            break

    # After training, load the best model and test
    if args.local_rank in [-1, 0]:
        logger.info("Best Validation Accuracy: %f" % best_acc)
        logger.info("Loading the best model saved during validation...")
        
        # Load the model checkpoint
        dir = os.path.join(args.output_dir, f"{args.name}_fold{fold}_checkpoint_with_weight.bin")
        checkpoint = torch.load(dir)
        print(checkpoint.keys())  # This will show you the available keys in the checkpoint file
        
        # Load model state
        model.load_state_dict(checkpoint)
        print(f"Checkpoint loaded from {args.output_dir}")

        # Close the writer (if needed)
        writer.close()
    

    # Conduct testing across all processes
    logger.info("Starting testset evaluation...")
    test_accuracy, patches_names, true_labels, all_Altertnative_labels, predictions, top_patches_probabilities, wsi_names, patches_indices, patches_coordinates = test(model, test_loader)
    _, tune_patches_names, tune_true_labels, tune_Altertnative_labels, tune_predictions, tune_top_patches_probabilities, tune_wsi_names, tune_patches_indices, tune_patches_coordinates = test(model, tune_loader)
    #Log the test Accuracy
    logger.info("Test Accuracy: %f" % test_accuracy)
    logger.info("End of the Training and testing!")
    # Capture end time
    end_time = time.time()  # End timing
    total_duration = end_time - start_time  # Calculate total duration
    

    logger.info("Total duration of training and testing: %f seconds" % total_duration)

    logger.info(f"Tuning the threshold for the training folds for fold {fold}")


    # Build a DataFrame of ALL train+val patches (the "tune" set)
    patch_df = pd.DataFrame({
    "Patch_Name":                        tune_patches_names,
    "True_patch_Label":                  tune_true_labels,
    "Prediction":                        tune_predictions,
    "Top_probability":                   tune_top_patches_probabilities,
    "WSI_Name":                          tune_wsi_names,
    "Patch_Index":                       tune_patches_indices,
    "Coordinate":                        tune_patches_coordinates
    })
    
    # --- new: save this fold's tune‐set predictions ---
    tune_dir = os.path.join(args.output_dir, "tune_predictions")
    os.makedirs(tune_dir, exist_ok=True)
    csv_path = os.path.join(tune_dir, f"fold{fold}_patch_level_tune_predictions.csv")
    patch_df.to_csv(csv_path, index=False)
    logger.info(f"[Fold {fold}] Saved tune‐set predictions to: {csv_path}")
       
    # Define your threshold grid
    ths = np.arange(0.05, 0.64, 0.01)
    ths2 = np.arange(0.05, 0.27, 0.01)
    # ths3 = np.arange(0.05, 0.15, 0.01)

    # Prepare storage for best (accuracy, threshold) per method
    best = {
        "majority": (0.0, 0.0),
        "softmax":  (0.0, 0.0)
        # "gradcam":  (0.0, 0.0)
    }

    # 1) Majority voting
    for t in ths:
        y_true, y_pred = [], []
        for wsi, grp in patch_df.groupby("WSI_Name"):
            true_lbl = int(grp["True_patch_Label"].any())
            pred_lbl = int(grp["Prediction"].mean() >= t)
            y_true.append(true_lbl)
            y_pred.append(pred_lbl)
        acc = float((np.array(y_true) == np.array(y_pred)).mean())
        old_acc, old_t = best["majority"]
        if acc > old_acc or (acc == old_acc and t > old_t):
            best["majority"] = (acc, t)

    # 2) Softmax-weighted voting
    for t in ths2:
        y_true, y_pred = [], []
        for wsi, grp in patch_df.groupby("WSI_Name"):
            true_lbl = int(grp["True_patch_Label"].any())
            w        = grp["Top_probability"].to_numpy()
            fusion   = np.where(grp["Prediction"] == 1, 1, -1) * w
            pred_lbl = int(fusion.mean() > t)
            y_true.append(true_lbl)
            y_pred.append(pred_lbl)
        acc = float((np.array(y_true) == np.array(y_pred)).mean())
        old_acc, old_t = best["softmax"]
        if acc > old_acc or (acc == old_acc and t > old_t):
            best["softmax"] = (acc, t)



    # Extract the best thresholds
    fold_thresholds = {
        "majority": best["majority"][1]
    }

    # 2) Also save this fold’s thresholds into the same tune directory
    thr_path = os.path.join(tune_dir, f"fold{fold}_thresholds.json")
    with open(thr_path, "w") as jf:
        json.dump(fold_thresholds, jf)
    logger.info(f"[Fold {fold}] Saved thresholds to: {thr_path}")


    # Log them
    logger.info(f"Fold {fold} thresholds → {fold_thresholds}")

    return train_epoch_losses, train_losses, val_losses, val_accuracies, best_acc, test_accuracy, patches_names, true_labels,all_Altertnative_labels, predictions, top_patches_probabilities, wsi_names, patches_indices, patches_coordinates, fold_thresholds
 
