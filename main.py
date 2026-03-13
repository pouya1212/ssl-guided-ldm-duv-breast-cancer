# coding=utf-8
from __future__ import absolute_import, division, print_function
import time
import logging
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, ConcatDataset
import torch.nn.functional as F  # for softmax computation

from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
import torch.distributed as dist
import json
import itertools
import seaborn as sns
from itertools import cycle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import pandas as pd
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
# from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from PIL import Image
from collections import defaultdict, Counter  # Import Counter along with defaultdict
from PIL import Image, ImageDraw
import re
from pyfiglet import Figlet
logger = logging.getLogger(__name__) # The logging module allows you to configure how messages are logged, where they are stored, and how they are formatted.

Image.MAX_IMAGE_PIXELS = None  # Disables the decompression bomb check

############################################################################################################################################################
# dataset class for delivering patches of WSI images along with their names, WSI name, index of patch, and its corresponding coordinates on the WSI image. 
patch_csv_weights1 = '/data/users4/pafshin1/My_Projects/DATA1/Dataset_1-matched-breast-corrected/INPUT_PATCHES/Updated_metadata_patches.csv'
patch_dir1 = '/data/users4/pafshin1/My_Projects/DATA1/Dataset_1-matched-breast-corrected/INPUT_PATCHES'
patchdataset1 = TumorImageDataset(
            csv_file=patch_csv_weights1,
            root_dir=patch_dir1,
            resize_size=(224, 224), transform=True)
    
patch_csv_weights2 = '/data/users4/pafshin1/My_Projects/DATA2/Dataset_2-matched-breast-corrected/INPUT_PATCHES/Updated_metadata_patches.csv'
patch_dir2 = '/data/users4/pafshin1/My_Projects/DATA2/Dataset_2-matched-breast-corrected/INPUT_PATCHES'
patchdataset2 = Tumor_Image_Patch_Dataset(
            csv_file=patch_csv_weights2,
            root_dir=patch_dir2,
            resize_size=(224, 224), transform=True)

meta1 = pd.read_csv(patch_csv_weights1)
meta2 =  pd.read_csv(patch_csv_weights2)

synthetic_both_classes = SyntheticTumorImageDataset(
    benign_dir="/data/users4/pafshin1/Diffusion Models/Large-Image-Diffusion/Output_Synthesis_Best/Benign/Benign_ssl_vq_4_eta_0.0_scale_2.0_combined",
    malignant_dir="/data/users4/pafshin1/Diffusion Models/Large-Image-Diffusion/Output_Synthesis_Best/Malignant/Malignant_ssl_vq_4_eta_0.0_scale_2.0_combined",
    mode="both"
)   

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

############################################################################################################################
# function for keep track of taining loss, etc
def save_plots_for_fold(fold, output_dir, train_losses, train_epoch_losses, val_losses, val_accuracies):
    """
    Save the training and validation loss/accuracy plots for each fold.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.xlabel("Total number of Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_loss_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

    # Plot training average loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_losses, label='Average Loss', color='blue', marker='o', linestyle='-')
    plt.title(f'Average Loss Over Training Steps - Fold {fold}')
    plt.xlabel('Epochs (each point corresponds to avg loss over 50 training steps = 1 epoch )')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"average_loss_plot_fold{fold}.png"))  # Save with fold info
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epochs (each point corresponds to evaluation after 50 training steps = 1 epoch)")
    plt.ylabel("Loss")
    plt.title(f"Validation Loss - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_loss_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

    # Plot validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epochs (each point corresponds to evaluation after 50 training steps = 1 epoch)")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_accuracy_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

#################################################################################################################################
# function that generates box plots if args, all patch_names, all patch_labels, and their predictions, and all WSI image names will be provided to it
#overlay patches predictions on WSIs
def visualize_patch_locations(args, all_patch_names, all_patch_labels, all_patch_predictions, all_wsi_names, all_patch_coordinates, patch_size=400):
    # Create output folder for box plots
    box_plot_dir = os.path.join(args.output_dir, "Box_Plot_Images")
    os.makedirs(box_plot_dir, exist_ok=True)

    # Dictionary to store WSI-level data
    wsi_data = defaultdict(list)

    # Group patches by their corresponding WSI image
    for patch_name, wsi_name, label, prediction, coordinates in zip(all_patch_names, all_wsi_names, all_patch_labels, all_patch_predictions, all_patch_coordinates):
        # Extract coordinates from the patch name
        pattern = r'_(\d+)_(\d+)\.tif$'
        match = re.search(pattern, patch_name)

        if match:
            y_coord, x_coord = int(match.group(1)), int(match.group(2))
            wsi_data[wsi_name].append((patch_name, x_coord, y_coord, label, prediction))

    csv_data = []  # To store per-patch metadata

    # Iterate over each WSI
    for wsi_name, patches in wsi_data.items():
        # Build possible paths in both batch1 and batch2 directories
        possible_paths = [
            os.path.join(args.wsi_img_path1, f"{wsi_name}.tif"),
            os.path.join(args.wsi_img_path1, f"{wsi_name}.jpg"),
            os.path.join(args.wsi_img_path2, f"{wsi_name}.tif"),
            os.path.join(args.wsi_img_path2, f"{wsi_name}.jpg"),
        ]

        # Pick the first existing path
        wsi_path = None
        for path in possible_paths:
            if os.path.exists(path):
                wsi_path = path
                break

        if wsi_path is None:
            print(f"[Warning] WSI image {wsi_name}.tif or .jpg not found in either directory.")
            continue

        print(f"[Found] WSI image: {wsi_path}")

        # Load WSI and prepare copies for drawing
        wsi_image = Image.open(wsi_path)
        predicted_image = wsi_image.copy()
        ground_truth_image = wsi_image.copy()

        draw_predicted = ImageDraw.Draw(predicted_image)
        draw_ground_truth = ImageDraw.Draw(ground_truth_image)

        for patch_name, x_coord, y_coord, true_label, pred_label in patches:
            box = [x_coord, y_coord, x_coord + patch_size, y_coord + patch_size]

            # Predicted
            pred_color = "green" if pred_label == 0 else "red"
            draw_predicted.rectangle(box, outline=pred_color, width=15)

            # Ground Truth
            true_color = "green" if true_label == 0 else "red"
            draw_ground_truth.rectangle(box, outline=true_color, width=15)

            # CSV metadata
            csv_data.append({
                'WSIName': wsi_name,
                'PatchName': patch_name,
                'X_Coordinate': x_coord,
                'Y_Coordinate': y_coord,
                'TrueLabel': true_label,
                'Prediction': pred_label,
                'WSI_Path': wsi_path   # ✅ keep record of which dir it came from
            })

        # Save visualization images
        predicted_image.save(os.path.join(box_plot_dir, f"{wsi_name}_predicted.jpg"))
        ground_truth_image.save(os.path.join(box_plot_dir, f"{wsi_name}_ground_truth.jpg"))

    # Save CSV file with all patches info
    results_dir = os.path.join(args.output_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    patch_csv_path = os.path.join(results_dir, "all_patches_information.csv")
    pd.DataFrame(csv_data).to_csv(patch_csv_path, index=False)

    print("\n✅ Patch locations visualized and saved for all WSIs.")
    print("📁 Box plots saved in:", box_plot_dir)
    print("📄 Patch info CSV saved to:", patch_csv_path)


##################################################################################################################################

def summarize(metric_list, name, method, f):
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    print(f"{method:<18} | {name:<12}: {mean:.4f} ± {std:.4f}")
    f.write(f"{method:<18} | {name:<12}: {mean:.4f} ± {std:.4f}\n")

#################################################################################################################################
# define a main function, including all args, which:
# 1) applies n-fold cross-validation, saves the Train, val loss, and accuracy along with test accuracy
# 2) It will take the mean and standard_deviation of the different folds' accuracy
# 3) It will capture the patches' information for each fold, such as the name of the WSI image, the patch name, the index, and its coordinate
# 4) It will then use this information to compute majority voting accuracy for the WSI image and compare the prediction with the label of the WSI image.    
# 5) It will generate a box plot on the WSI image using the prediction of wsi' patches 

   
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", type=str, default="Breast corrected Combined",
                    help="Name of this run. Used for monitoring.")
    # parser.add_argument("--dataset", choices=["tumor"], default="tumor",
    #                     help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/data/users4/pafshin1/Implementation/Vision Transformer/Original_VIT2/models/Pretrained_models/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="WSI_Classification_corrected_test_combined_without_synth_new_folds4", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--wsi_img_path1', type=str, default='/data/users4/pafshin1/My_Projects/DATA1/Dataset_1-matched-breast-corrected/INPUT_WSI',  # Default value 
                        help='Path to the directory containing WSI images')
    parser.add_argument('--wsi_img_path2', type=str, default='/data/users4/pafshin1/My_Projects/DATA2/Dataset_2-matched-breast-corrected/INPUT_WSI',  # Default value 
                        help='Path to the directory containing WSI images')                    
    parser.add_argument('--wsi_csv_folds1', type=str, default='/data/users4/pafshin1/My_Projects/DATA1/Dataset_1-matched-breast-uncorrected/WSI_Labels.csv',  # Default value 
                        help='Path to the directory csv for generating folds')
    parser.add_argument('--wsi_csv_folds2', type=str, default='/data/users4/pafshin1/My_Projects/DATA2/Dataset_2-matched-breast-uncorrected/wsi_level_labels.csv',  # Default value 
                        help='Path to the directory csv for generating folds')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=40, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for. I considered the  5% total_steps which is 5000 = 250")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_epochs", default=40, type=int, help="number of epochs.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1: # non-distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend, which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))



    # Set seed
    set_seed(args)
    
    ####################################################################################
    # Example usage:
    #here we defined a list of manual folds for 1st batch of the dataset
    manualFolds1 = []
    #some wsis were excluded 
    Extra_WSIs =[]

    # Make a deep copy
    folds_updated = [fold.copy() for fold in manualFolds1]

    # Add extras in round-robin
    fold_iter = cycle(range(len(folds_updated)))
    for wsi, fold_idx in zip(Extra_WSIs, fold_iter):
        folds_updated[fold_idx].append(wsi)

    folds1 = folds_updated 
    # Print number of WSIs per fold
    for i, fold in enumerate(folds1 , start=1):
        print(f"Fold {i}: {fold} WSIs")


    fancy_title = Figlet(font='standard')
    
    # Banner lines to be displayed and logged
    lines = [
        "WSI-Level and Patch-Level Classification for",
        args.name,
        "Batch1 (Training Both Batches)",
        "Vision Transformer + Majority Voting",
        "Pouya Afshin - GSU"
    ]

    wsidata_df1 = pd.read_csv(args.wsi_csv_folds1)

    wsidata_df2 = pd.read_csv(args.wsi_csv_folds2)
    
   
    
    
    train_epoch_losses_per_fold = []
    train_losses_per_fold = []
    val_losses_per_fold = []
    val_accuracies_per_fold = []
    test_accuracies_per_fold = []
    best_accuracies_per_fold = []

    all_test_indices = []


    # Initialize lists to store patch-level data across all folds
    
    all_patch_names = []
    all_patch_labels = []
    all_patches_alternative_labels = []
    all_patch_predictions = []
    all_wsi_names = []
    all_patch_indices = []
    all_patch_coordinates = []
    
    # Initialize a list to store top probabilities for weighted voting
    all_top_patches_probabilities = []

    sensitivity_per_fold = []
    specificity_per_fold = []
    
    all_fold_thresholds = []
    
    all_slide_dfs = []


    ################WSI-LEVEL test evaluation######################

    # Majority voting metrics per fold
    maj_accs, maj_sens, maj_specs, maj_precs, maj_f1s = [], [], [], [], []

    # Softmax-weighted voting metrics per fold
    soft_accs, soft_sens, soft_specs, soft_precs, soft_f1s = [], [], [], [], []

    ####################################################################

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    results_file_path = os.path.join(results_dir, "cross_validation_results.txt")

    
    # Open a text file to save results for each fold
    with open(results_file_path, "w") as f:
        for line in lines:
            banner = fancy_title.renderText(line)
            print(banner)
            logger.info("\n" +banner)
            f.write(banner + "\n")
            f.flush()
            os.fsync(f.fileno())
        
        
        # WSI-Level Distribution 
        counts_wsi2 = wsidata_df2['Binary_Label'].value_counts().sort_index()
        percentages2 = counts_wsi2 / counts_wsi2.sum() *100 
        wsi_distribution_msg2 = (
            "\n WSI-level Distribution : \n"
            f"Benign (0):   {counts_wsi2[0]} patches ({percentages2[0]:.2f}%)\n"
            f"Malignant (1) :{counts_wsi2[1]} patches ({percentages2[1]:.2f}%)\n"
        )
        print(wsi_distribution_msg2)
        logger.info(wsi_distribution_msg2)
        f.write(wsi_distribution_msg2 + "\n")
        f.flush()
        os.fsync(f.fileno())
        #================save plot=============
        labels = ['Benign (0)', 'Malignant (1)']
        plt.figure(figsize = (5,6))
        plt.bar(labels, counts_wsi2, color= ['green', 'red'])
        plt.title('WSI-Level Distribution')
        plt.ylabel('Number of WSIs')
        plt.xlabel('Categories')
        max_count = max(counts_wsi2)
        plt.ylim(0, max_count*1.15) 

        for i, (count, pct) in enumerate(zip(counts_wsi2, percentages2)):
            plt.text(i, count/2, f"{count}", ha = 'center', va ='center', color ='white', fontsize =12, fontweight ='bold' )

            #percentage outside the bar
            plt.text(i, count + max_count * 0.02, f"{pct:.1f}%", ha= 'center', va='center', fontsize='12', fontweight ='bold')
        
        plot_path = os.path.join(results_dir, "wsi-level_distribution_breast_corrected_unadjusted_4x_10x_comparison.png")
        plt.savefig(plot_path, bbox_inches='tight')

        logger.info(f"Saved WSI-Level distribution plot to: {plot_path}")
        f.write(f"Plot saved to: {plot_path}\n")

        f.flush()
        os.fsync(f.fileno())

        #here provided the 2nd list of folds for  WSIs (we used two different batch of dataset)
        folds2 = []

        numFolds = len(folds2)
        #============Patch-Level Distribution after filtering===========
        patch_counts2 = meta2['Binary_Label'].value_counts().sort_index()
        patch_percentages2 = patch_counts2 / patch_counts2.sum() * 100 

        patch_distribution_msg2 = (
            "\n Patch-level Distribution : \n"
            f"Benign (0):   {patch_counts2[0]} patches ({patch_percentages2[0]:.2f}%)\n"
            f"Malignant (1) :{patch_counts2[1]} patches ({patch_percentages2[1]:.2f}%)\n"
        )
        print(patch_distribution_msg2)
        logger.info(patch_distribution_msg2)
        f.write(patch_distribution_msg2 + "\n")
        f.flush()
        os.fsync(f.fileno())

        #====================save Patch-level distribution bar plot=====================

        patch_labels = ['Benign (0)', 'Malignant (1)']
        plt.figure(figsize = (5,6))
        plt.bar(patch_labels, patch_counts2, color= ['green', 'red'])
        plt.title('Patch-Level Distribution')
        plt.ylabel('Number of patches')
        plt.xlabel('Categories')
        max_count = max(patch_counts2)
        plt.ylim(0, max_count*1.15) 

        for i, (count, pct) in enumerate(zip(patch_counts2, patch_percentages2)):
            plt.text(i, count/2, f"{count}", ha = 'center', va ='center', color ='white', fontsize =12, fontweight ='bold' )

            #percentage outside the bar
            plt.text(i, count + max_count*0.02, f"{pct:.1f}%", ha= 'center', va='center', fontsize='12', fontweight ='bold')

        plot_path2 = os.path.join(results_dir, "patch-level_distribution_breast_corrected_unadjusted_4x_10x_comparison.png")
        plt.savefig(plot_path2, bbox_inches='tight')

        logger.info(f"Saved Patch-Level distribution plot to: {plot_path2}")
        f.write(f"Plot saved to: {plot_path2}\n")

        f.flush()
        os.fsync(f.fileno())

        ###############################Total WSI-Level Distribution for batch1 +batch2 #####################################
        # WSI-Level Distribution 
        # Ensure both labels exist in both counts (in case one dataset is missing a class)
        counts_wsi1 = wsidata_df1['Binary_Label'].value_counts().sort_index()
        counts_wsi1 = counts_wsi1.reindex([0,1], fill_value=0)
        counts_wsi2 = counts_wsi2.reindex([0,1], fill_value=0)

        # Totals
        counts_total = counts_wsi1 + counts_wsi2
        total_benign_wsis = counts_total[0]
        total_malignant_wsis = counts_total[1]

        # Percentages
        percentages = counts_total / counts_total.sum() * 100

        # Message
        wsi_distribution_total = (
            "\n WSI-level Distribution:\n"
            f"Benign (0):    {total_benign_wsis} WSIs ({percentages[0]:.2f}%)\n"
            f"Malignant (1): {total_malignant_wsis} WSIs ({percentages[1]:.2f}%)\n"
        )
        print(wsi_distribution_total)

        #================ Plot ================
        #================ Plot ================
        labels = ['Benign (0)', 'Malignant (1)']
        plt.figure(figsize=(5,6))
        bars = plt.bar(labels, counts_total, color=['green', 'red'])
        plt.title('WSI-Level Distribution')
        plt.ylabel('Number of WSIs')
        plt.xlabel('Categories')

        max_count = counts_total.max()
        plt.ylim(0, max_count*1.25)  # leave extra space for percentages

        # Annotate numbers inside bars + percentages on top
        for i, (count, pct) in enumerate(zip(counts_total, percentages)):
            # Number inside the bar
            plt.text(i, count/2, f"{count}", ha='center', va='center',
                    color='white', fontsize=12, fontweight='bold')
            
            # Percentage above the bar
            plt.text(i, count + max_count*0.03, f"{pct:.1f}%", ha='center',
                    va='bottom', fontsize=12, fontweight='bold')

        
        plot_path2 = os.path.join(results_dir, "wsi-level_distribution_batch1_batch2_4x_corrected.png")
        plt.savefig(plot_path2, bbox_inches='tight')

        logger.info(f"Saved Total WSI-Level distribution plot to: {plot_path2}")
        f.write(f"Plot saved to: {plot_path2}\n")

        f.flush()
        os.fsync(f.fileno())
        ###################################Total Patch-level Distribution#########################
        # 
        # Count benign (0) and malignant (1) patches
        patch_counts1 = meta1['Binary_Label'].value_counts().sort_index()
        # Ensure both classes exist
        patch_counts1 = patch_counts1.reindex([0,1], fill_value=0)
        patch_counts2 = patch_counts2.reindex([0,1], fill_value=0)

        # Combine
        patch_counts_total = patch_counts1 + patch_counts2
        patch_percentages = patch_counts_total / patch_counts_total.sum() * 100

        #================ Plot ================
        labels = ['Benign (0)', 'Malignant (1)']
        plt.figure(figsize=(5,6))
        bars = plt.bar(labels, patch_counts_total, color=['green', 'red'])
        plt.title('Patch-Level Distribution')
        plt.ylabel('Number of Patches')
        plt.xlabel('Categories')

        max_count = patch_counts_total.max()
        plt.ylim(0, max_count*1.25)

        # Annotate numbers inside bars + percentages on top
        for i, (count, pct) in enumerate(zip(patch_counts_total, patch_percentages)):
            # Number inside the bar
            plt.text(i, count/2, f"{count}", ha='center', va='center',
                    color='white', fontsize=12, fontweight='bold')
            
            # Percentage above the bar
            plt.text(i, count + max_count*0.03, f"{pct:.1f}%", ha='center',
                    va='bottom', fontsize=12, fontweight='bold')

        plot_path2 = os.path.join(results_dir, "total_patch-level_distribution_batch1_batch2_4x_corrected.png")
        plt.savefig(plot_path2, bbox_inches='tight')

        logger.info(f"Saved Total patch-Level distribution plot to: {plot_path2}")
        f.write(f"Plot saved to: {plot_path2}\n")

        f.flush()
        os.fsync(f.fileno())


        f.write("=== Cross-Validation Results ===\n")
        f.flush()
        os.fsync(f.fileno())

        for foldNum in tqdm(range(0, numFolds), desc='Patch Classification for cross-validation', leave=True):
            print(" Starting Fold:", foldNum+1)

            ##############################Batch1####################################
            # Train on all folds except the one specified (current fold is for testing)
            trainfolds1 = np.concatenate(folds1[:foldNum] + folds1[foldNum+1:])  # Select all patches except the current fold
            
            trainfolds1 = [wsi for wsi in trainfolds1]
            # print(trainfolds)
            train_val_fold_indices1 = meta1[meta1['WSI'].isin(trainfolds1)].index.tolist()
            # Adjust for 1-based indexing by adding 2
            train_val_fold_indices1 = [idx  for idx in train_val_fold_indices1]

            # # Select the current fold as the test fold
            test_fold1 = folds1[foldNum]
            test_fold1 = [wsi for wsi in test_fold1]
            print(test_fold1)
            testfold_indices1 = meta1[meta1['WSI'].isin(test_fold1)].index.tolist()
            testfold_indices1 = [idx  for idx in testfold_indices1]

            ##############Making dataset and dataloader for training, validation and testset using folds indices########
            # Split the train+val set into training and validation sets, alocate 20 percent of the indices for validation and the remaining for training
            
            
            val_split1 = 0.2
            train_size1 = int((1 - val_split1) * len(train_val_fold_indices1))
            all_trainin_validation_set1 = Subset(patchdataset1, train_val_fold_indices1)
            len(all_trainin_validation_set1)
            np.random.shuffle(train_val_fold_indices1)  # Shuffle train+val to randomize training/validation split
            train_indices1 = train_val_fold_indices1[:train_size1]
            val_indices1 = train_val_fold_indices1[train_size1:]
            
            
            train_set1 = Subset(patchdataset1, train_indices1)
            len(train_set1)

            val_set1 = Subset(patchdataset1, val_indices1)
            len(train_set1)
            
            test_set1 = Subset(patchdataset1, testfold_indices1)
            len(test_set1)
            
        
            
            # Debugging outputs
            # Debugging outputs
            print("train_folds of 1st Batch:", trainfolds1)
            print("test_fold of 1st Batch:", test_fold1)
            print("training set of 1st Batch length :", len(train_set1))
            print("Validation set of 1st Batch length :", len(val_set1))
            print("test set of 1st Batch length :", len(test_set1))
            print("all validation + training patches of 1st Batch:", len(train_set1)+len(val_set1))
            print(" The total number of patches of 1st Batch used for training and evaluation :", len(train_set1)+len(val_set1)+len(test_set1))
            print("The total WSI images of 1st Batch used for training and evaluation:", len(trainfolds1) +len(test_fold1))
            # print("list of test indices:", testfold_indices)
            
            f.write(f"train_folds of 1st Batch: {trainfolds1}\n")
            f.write(f"test_fold of 1st Batch:   {test_fold1}\n")
            f.write(f"WSIs train/val of 1st Batch: {len(trainfolds1)}\n")
            f.write(f"WSIs test of 1st Batch:      {len(test_fold1)}\n")
            f.write(f"total training patches of 1st Batch:   {len(train_set1)}\n")
            f.write(f"total validation patches oof 1st Batch: {len(val_set1)}\n")
            f.write(f"total test patches of of 1st Batch:       {len(test_set1)}\n")
            f.flush()
            os.fsync(f.fileno())
            ##############################batch2######################################

            # Train on all folds except the one specified (current fold is for testing)
            trainfolds2 = np.concatenate(folds2[:foldNum] + folds2[foldNum+1:])  # Select all patches except the current fold
            
            trainfolds2 = [wsi for wsi in trainfolds2]
            # print(trainfolds)
            train_val_fold_indices2 = meta2[meta2['WSI'].isin(trainfolds2)].index.tolist()
            # Adjust for 1-based indexing by adding 2
            train_val_fold_indices2 = [idx  for idx in train_val_fold_indices2]

            # # Select the current fold as the test fold
            test_fold2 = folds2[foldNum]
            test_fold2 = [wsi for wsi in test_fold2]
            print(test_fold2)
            testfold_indices2 = meta2[meta2['WSI'].isin(test_fold2)].index.tolist()
            testfold_indices2 = [idx  for idx in testfold_indices2]

            
            ##############Making dataset and dataloader for training, validation and testset using folds indices########
            # Split the train+val set into training and validation sets, alocate 20 percent of the indices for validation and the remaining for training
            
            val_split2 = 0.2
            train_size2 = int((1 - val_split2) * len(train_val_fold_indices2))
            all_trainin_validation_set2 = Subset(patchdataset2, train_val_fold_indices2)
            len(all_trainin_validation_set2)
            np.random.shuffle(train_val_fold_indices2)  # Shuffle train+val to randomize training/validation split
            train_indices2 = train_val_fold_indices2[:train_size2]
            val_indices2 = train_val_fold_indices2[train_size2:]
            
            
            train_set2 = Subset(patchdataset2, train_indices2)


            train_set = ConcatDataset([train_set2, train_set1]) # combination of two real dataset and one synthetic

            

            len(train_set)
            val_set2 = Subset(patchdataset2, val_indices2)
            len(train_set2)
            
            test_set2 = Subset(patchdataset2, testfold_indices2)
            len(test_set2)
            
            all_trainin_validation_set = ConcatDataset([all_trainin_validation_set2, all_trainin_validation_set1])
            len(all_trainin_validation_set)
            val_set = ConcatDataset([val_set2, val_set1])

            test_set = ConcatDataset([test_set2, test_set1])

            # Debugging outputs
            # Debugging outputs
            print("train_folds of 2nd Batch:", trainfolds2)
            print("test_fold of 2nd Batch:", test_fold2)
            print("training set of 2nd Batch length :", len(train_set2))
            print("Validation set of 2nd Batch length :", len(val_set2))
            print("test set of 2nd Batch length :", len(test_set2))
            print("all validation + training patches of 2nd Batch:", len(train_set2)+len(val_set2))
            print(" The total number of patches of 2nd Batch used for training and evaluation :", len(train_set2)+len(val_set2)+len(test_set2))
            print("The total WSI images of 2nd Batch used for training and evaluation:", len(trainfolds2) +len(test_fold2))
            # print("list of test indices:", testfold_indices)
            
            f.write(f"train_folds of 2nd Batch: {trainfolds2}\n")
            f.write(f"test_fold of 2nd Batch:   {test_fold2}\n")
            f.write(f"WSIs train/val of 2nd Batch: {len(trainfolds2)}\n")
            f.write(f"WSIs test of 2nd Batch:      {len(test_fold2)}\n")
            f.write(f"total training patches of 2nd Batch:   {len(train_set2)}\n")
            f.write(f"total validation patches oof 2nd Batch: {len(val_set2)}\n")
            f.write(f"total test patches of of 2nd Batch:       {len(test_set2)}\n")
            f.flush()
            os.fsync(f.fileno())
            
            # Debugging outputs
            #Total batch1 + batch2
            
            print("Total training set of batch1 + batch2 length :", len(train_set))
            print("Total validation set of batch1 + batch2 length :", len(val_set))
            print("Total test set of batch1 +batch2  length :", len(test_set))
            print("Total validation + training patches of  batch1 +batch:", len(train_set)+len(val_set))
            print(" The total number of patches of  batch1 +batch for training and evaluation :", len(train_set)+len(val_set)+len(test_set))
            # print("list of test indices:", testfold_indices)
            
            
            f.write(f"Total training set of batch1 + batch2 length:   {len(train_set)}\n")
            f.write(f"Total validation set of batch1 + batch2 length: {len(val_set)}\n")
            f.write(f"Total test set of batch1 +batch2  length:       {len(test_set)}\n")
            f.flush()
            os.fsync(f.fileno())

            # Model & Tokenizer Setup
            args, model = setup(args)  # Initialize the model for each fold
            
            # Get DataLoader for train, val, and test sets
            train_loader, val_loader, test_loader, tune_loader = get_loader(args, train_set, val_set, test_set, all_trainin_validation_set)
            
            # Train and evaluate
            train_epoch_losses, train_losses, val_losses, val_accuracies, best_acc, test_accuracy, patches_names, true_labels, all_Altertnative_labels, predictions, top_patches_probabilities, wsi_names, patches_indices, patches_coordinates, fold_thresholds = train_test(
                args, model, train_loader, val_loader, test_loader, tune_loader, fold = foldNum + 1
            )
            
             # Store this fold’s thresholds
            all_fold_thresholds.append({
                "fold":     foldNum + 1,
                "majority": fold_thresholds["majority"],
                "softmax":  fold_thresholds["softmax"]
                # "gradcam":  fold_thresholds["gradcam"]
            })

            # After all folds, save them to CSV
            # Save after each append (or once after the loop)
            th_df = pd.DataFrame(all_fold_thresholds)
            csv_path = os.path.join(results_dir, "all_fold_thresholds.csv")
            th_df.to_csv(csv_path, index=False)

            f.write(f"\nSaved all-fold thresholds to: {csv_path}\n")
            f.flush(); os.fsync(f.fileno())

            print("Per‑fold thresholds:\n", th_df)


             # 1) Build a patch‑level DataFrame for this fold’s test set
            test_df = pd.DataFrame({
                "Patch_Name":                        patches_names,
                "True_patch_Label":                  true_labels,
                "True_Alternative_patch_Label":      all_Altertnative_labels,
                "Prediction":                        predictions,
                "Top_probability":                   top_patches_probabilities,
                "WSI_Name":                          wsi_names,
                "Patch_Index":                       patches_indices,
                "Coordinate":                        patches_coordinates,
            })

            #  Persist it under results_dir
            csv_path = os.path.join(results_dir, f"fold{foldNum+1}_patch_level_test.csv")
            test_df.to_csv(csv_path, index=False)

            # Optional logging
            f.write(f"[Fold {foldNum+1}] Saved patch‑level test results to: {csv_path}\n")
            f.flush(); os.fsync(f.fileno())

            # Extract true labels and predictions for the test fold
            true_labels = np.array(true_labels)  # Ground truth for patches
            predictions = np.array(predictions)  # Predictions for patches
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero
            
            # Append to lists
            sensitivity_per_fold.append(sensitivity)
            specificity_per_fold.append(specificity)
            
            # Log fold-specific sensitivity and specificity
            f.write(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")
            print(f"Fold {foldNum + 1} Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
            

            # Read in the thresholds just computed
            maj_th  = fold_thresholds["majority"]

            # 2) Build slide‑level rows exactly as before, but *without* any grid search:
            rows = []
            for wsi, grp in test_df.groupby("WSI_Name"):
                # ground‑truth WSI label = 1 if any patch is malignant
                true_lbl   = int(grp["True_patch_Label"].any())

                # majority‐vote prediction
                maj_pred   = int(grp["Prediction"].mean() >= maj_th)


                rows.append({
                    "WSI_Name":  wsi,
                    "True_Label": true_lbl,
                    "Maj_Pred":   maj_pred,
                    # "Grad_Pred":  grad_pred
                })

            # 3) Turn it back into a DataFrame just like before
            slide_df = pd.DataFrame(rows)
            
            ##compute Testfold accuracy, sensitivity, specificity, F1, precision
            # Extract true labels
            y_true = slide_df["True_Label"].to_numpy()

            # --- Majority Voting ---
            y_pred_maj = slide_df["Maj_Pred"].to_numpy()
            acc_maj, sens_maj, spec_maj, prec_maj, f1_maj = compute_wsi_metrics(y_true, y_pred_maj)

            # --- Softmax Voting ---
            y_pred_soft = slide_df["Soft_Pred"].to_numpy()
            acc_soft, sens_soft, spec_soft, prec_soft, f1_soft = compute_wsi_metrics(y_true, y_pred_soft)

            f.write("\n=== WSI-Level Metrics ===\n")

            print("=== WSI-Level Metrics ===\n")
            print("--- Majority Voting ---")
            print(f"Accuracy:    {acc_maj:.4f}")
            print(f"Sensitivity: {sens_maj:.4f}")
            print(f"Specificity: {spec_maj:.4f}")
            print(f"Precision:   {prec_maj:.4f}")
            print(f"F1 Score:    {f1_maj:.4f}")

            print("\n--- Softmax Voting ---")
            print(f"Accuracy:    {acc_soft:.4f}")
            print(f"Sensitivity: {sens_soft:.4f}")
            print(f"Specificity: {spec_soft:.4f}")
            print(f"Precision:   {prec_soft:.4f}")
            print(f"F1 Score:    {f1_soft:.4f}")

            

            f.write("\n-- Majority Voting --\n")
            f.write(f"Accuracy:    {acc_maj:.4f}\n")
            f.write(f"Sensitivity: {sens_maj:.4f}\n")
            f.write(f"Specificity: {spec_maj:.4f}\n")
            f.write(f"Precision:   {prec_maj:.4f}\n")
            f.write(f"F1 Score:    {f1_maj:.4f}\n")

            f.write("\n-- Softmax Voting --\n")
            f.write(f"Accuracy:    {acc_soft:.4f}\n")
            f.write(f"Sensitivity: {sens_soft:.4f}\n")
            f.write(f"Specificity: {spec_soft:.4f}\n")
            f.write(f"Precision:   {prec_soft:.4f}\n")
            f.write(f"F1 Score:    {f1_soft:.4f}\n")

            # Ensure write is committed to disk
            f.flush()
            os.fsync(f.fileno())

            # Store fold-level metrics for aggregation later
            maj_accs.append(acc_maj)
            maj_sens.append(sens_maj)
            maj_specs.append(spec_maj)
            maj_precs.append(prec_maj)
            maj_f1s.append(f1_maj)

            soft_accs.append(acc_soft)
            soft_sens.append(sens_soft)
            soft_specs.append(spec_soft)
            soft_precs.append(prec_soft)
            soft_f1s.append(f1_soft)

            # 4) (Optional) Save & log
            slide_csv = os.path.join(results_dir, f"fold{foldNum+1}_wsi_level_preds.csv")
            slide_df.to_csv(slide_csv, index=False)
            f.write(f"[Fold {foldNum+1}] Saved slide‑level predictions to: {slide_csv}\n")
            f.flush(); os.fsync(f.fileno())

            all_slide_dfs.append(slide_df)

            # Store results for this fold
            train_epoch_losses_per_fold.append(train_epoch_losses)  # List of losses per epoch
            train_losses_per_fold.append(train_losses)  # Final train loss per epoch
            val_losses_per_fold.append(val_losses)  # Validation losses per epoch
            val_accuracies_per_fold.append(val_accuracies)  # Validation accuracies per epoch
            test_accuracies_per_fold.append(test_accuracy)  # Single test accuracy value
            best_accuracies_per_fold.append(best_acc)  # Best validation accuracy
            
            
            # for wsi classification stores the information and predicition for each patch
            
             # Append patch-level data for this fold to the aggregated lists
            all_patch_names.extend(patches_names)
            all_patch_labels.extend(true_labels)
            all_patches_alternative_labels.extend(all_Altertnative_labels)
            all_patch_predictions.extend(predictions)
            all_top_patches_probabilities.extend(top_patches_probabilities)  # Store higher probabilities coming out of softmax for patches at each fold
            all_wsi_names.extend(wsi_names)
            all_patch_indices.extend(patches_indices)
            all_patch_coordinates.extend(patches_coordinates)
            # all_patch_densenet_gradcam_importance_weights.extend(patch_densenet_gradcam_importance_weights)
            
            # Save fold-specific results to the text file
            f.write(f"\nFold {foldNum + 1} Results:\n")
            f.write(f"Train Epoch Losses: {train_epoch_losses}\n")
            f.write(f"Train Losses: {train_losses}\n")
            f.write(f"Validation Losses: {val_losses}\n")
            f.write(f"Validation Accuracies: {val_accuracies}\n")
            f.write(f"Best Validation Accuracy: {best_acc}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")
            f.flush()
            os.fsync(f.fileno())
            
            
            save_plots_for_fold(foldNum, results_dir, train_losses, train_epoch_losses, val_losses, val_accuracies)
            
             # Clear references
            del train_loader, val_loader, test_loader  # Delete DataLoaders
            del model  # Delete model

            torch.cuda.empty_cache()  # Clear cache again
        
        ####################################################################################################################
        
        # Once all folds are done, calculate averages and standard deviations for each metric

        # Convert lists to numpy arrays for easier computation
        # Convert lists to numpy arrays for easier computation
        sensitivity_per_fold = np.array(sensitivity_per_fold)
        specificity_per_fold = np.array(specificity_per_fold)
        
        train_epoch_losses_per_fold = np.array(train_epoch_losses_per_fold, dtype=object)
        train_losses_per_fold = np.array(train_losses_per_fold, dtype=object)
        val_losses_per_fold = np.array(val_losses_per_fold, dtype=object)
        val_accuracies_per_fold = np.array(val_accuracies_per_fold, dtype=object)
        test_accuracies_per_fold = np.array(test_accuracies_per_fold)
        best_accuracies_per_fold = np.array(best_accuracies_per_fold)
        
        

        # Calculate averages and standard deviations for each metric
        logger.info("Calculating averages and standard deviations for each metric")
        
        sensitivity_avg = np.mean(sensitivity_per_fold)
        sensitivity_std = np.std(sensitivity_per_fold)
        
        specificity_avg = np.mean(specificity_per_fold)
        specificity_std = np.std(specificity_per_fold)
        
        train_epoch_avg = np.mean([np.mean(fold) for fold in train_epoch_losses_per_fold])
        train_epoch_std = np.std([np.mean(fold) for fold in train_epoch_losses_per_fold])

        train_loss_avg = np.mean([np.mean(fold) for fold in train_losses_per_fold])
        train_loss_std = np.std([np.mean(fold) for fold in train_losses_per_fold])

        val_loss_avg = np.mean([np.mean(fold) for fold in val_losses_per_fold])
        val_loss_std = np.std([np.mean(fold) for fold in val_losses_per_fold])

        val_acc_avg = np.mean([np.mean(fold) for fold in val_accuracies_per_fold])
        val_acc_std = np.std([np.mean(fold) for fold in val_accuracies_per_fold])

        test_acc_avg = np.mean(test_accuracies_per_fold)
        test_acc_std = np.std(test_accuracies_per_fold)

        best_acc_avg = np.mean(best_accuracies_per_fold)
        best_acc_std = np.std(best_accuracies_per_fold)

        # Save final averages and standard deviations to the text file
        f.write("\n=== Overall Cross-Validation Results ===\n")
        f.write(f"Train Epoch Loss Average: {train_epoch_avg:.4f} ± {train_epoch_std:.4f}\n")
        f.write(f"Train Loss Average: {train_loss_avg:.4f} ± {train_loss_std:.4f}\n")
        f.write(f"Validation Loss Average: {val_loss_avg:.4f} ± {val_loss_std:.4f}\n")
        f.write(f"Validation Accuracy Average: {val_acc_avg:.4f} ± {val_acc_std:.4f}\n")
        f.write(f"Best Validation Accuracy Average: {best_acc_avg:.4f} ± {best_acc_std:.4f}\n")
        f.write(f"Test Accuracy Average: {test_acc_avg:.4f} ± {test_acc_std:.4f}\n")
        f.write(f"Sensitivity Average: {sensitivity_avg:.4f} ± {sensitivity_std:.4f}\n")
        f.write(f"Specificity Average: {specificity_avg:.4f} ± {specificity_std:.4f}\n")
        f.flush()
        os.fsync(f.fileno())
        
        # Print the values to the console
        print("\n=== Overall Cross-Validation Results ===")
        print(f"Train Epoch Loss Average: {train_epoch_avg:.4f} ± {train_epoch_std:.4f}")
        print(f"Train Loss Average: {train_loss_avg:.4f} ± {train_loss_std:.4f}")
        print(f"Validation Loss Average: {val_loss_avg:.4f} ± {val_loss_std:.4f}")
        print(f"Validation Accuracy Average: {val_acc_avg:.4f} ± {val_acc_std:.4f}")
        print(f"Best Validation Accuracy Average: {best_acc_avg:.4f} ± {best_acc_std:.4f}")
        print(f"Test Accuracy Average: {test_acc_avg:.4f} ± {test_acc_std:.4f}")
        print(f"Sensitivity Average: {sensitivity_avg:.4f} ± {sensitivity_std:.4f}")
        print(f"Specificity Average: {specificity_avg:.4f} ± {specificity_std:.4f}")
        
        logger.info(" End of Cross-Validation")
        ###################################################################################################
        ###################################################################################################
        # Add a header for WSI-level results
        
        # WSI majority voting Classification using the patches information and predictions and WSI true labels
        # Save patch-level data to CSV
        logger.info(" Whole Slide Image (WSI) Classification using Majority voting and weighted average voting Using Grad-CAM.")
        alternative_pred_map = {0: -1, 1: 1}  # Mapping 0 -> -1, 1 stays as 1
        
        # Create a list to store alternative predictions for each patch
        alternative_patch_predictions = [alternative_pred_map.get(pred, pred) for pred in all_patch_predictions]
        
        
        # 2) Concatenate into one big DataFrame (60 rows total)
        all_slides = pd.concat(all_slide_dfs, ignore_index=True)

        all_csv = os.path.join(results_dir, "all_60_wsi_level_preds.csv")
        all_slides.to_csv(all_csv, index=False)
        logger.info(f"Saved all‑fold WSI‑level predictions to: {all_csv}")

        # (Optional) save the full 60‑WSI CSV
        all_csv = os.path.join(results_dir, "all_60_wsi_level_preds.csv")
        all_slides.to_csv(all_csv, index=False)
        f.write(f"Saved concatenated slide‑level predictions to: {all_csv}\n")
        f.flush(); os.fsync(f.fileno())

        
        
        print(f"Length of Patch_Name: {len(all_patch_names)}")
        print(f"Length of True_patch_Label: {len(all_patch_labels)}")
        print(f"Length of True_Alternative_patch_Label: {len(all_patches_alternative_labels)}")
        print(f"Length of Prediction: {len(all_patch_predictions)}")
        print(f"Length of Alternative_prediction: {len(alternative_patch_predictions)}")
        print(f"Length of Top_probability: {len(all_top_patches_probabilities)}")
        print(f"Length of WSI_Name: {len(all_wsi_names)}")
        print(f"Length of Index: {len(all_patch_indices)}")
        print(f"Length of Coordinate: {len(all_patch_coordinates)}")
        # print(f"Length of Densenet Gradcam Patch Importance :{len(all_patch_densenet_gradcam_importance_weights)}")


        patch_data = pd.DataFrame({
            'Patch_Name': all_patch_names,
            'True_patch_Label': all_patch_labels,
            'True_Alternative_patch_Label' : all_patches_alternative_labels,
            'Prediction': all_patch_predictions,
            'Alternative_prediction': alternative_patch_predictions,
            'Top_probability' : all_top_patches_probabilities,
            # 'Densenet_Gradcam_Patch_Importance' : all_patch_densenet_gradcam_importance_weights,
            'WSI_Name': all_wsi_names,
            'Index': all_patch_indices,
            'Coordinate': all_patch_coordinates
        })
        
        
        # Save patch-level results to CSV in the output directory
        
        # 1) Save patch-level CSV
        patch_csv_path = os.path.join(results_dir, "patch_level_results.csv")
        patch_data.to_csv(patch_csv_path, index=False)
        f.write(f"\nSaved patch-level results to: {patch_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        #keep track of number of correct predictions/ wrong predicitons for each wsi
        wsi_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0})

        # Count correct and wrong predictions per WSI
        for _, row in patch_data.iterrows():
            wsi = row['WSI_Name']
            true_label = int(row['True_patch_Label'])
            pred_label = int(row['Prediction'])

            if pred_label == true_label:
                wsi_stats[wsi]['correct'] += 1
            else:
                wsi_stats[wsi]['wrong'] += 1

        # Prepare summary list
        summary_data = []
        for wsi, stats in wsi_stats.items():
            correct = stats['correct']
            wrong = stats['wrong']
            total = correct + wrong
            correct_pct = round((correct / total) * 100, 2) if total > 0 else 0.0
            wrong_pct = round((wrong / total) * 100, 2) if total > 0 else 0.0

            summary_data.append({
                'Sample': wsi,
                'Correct_Predictions': correct,
                'Wrong_Predictions': wrong,
                'Correct_Percentage': correct_pct,
                'Wrong_Percentage': wrong_pct,
                'Total_Patches': total
            })

        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(results_dir, "wsi_level_accuracy_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        
        f.write(f"\n✅ Saved WSI-level accuracy summary to: {summary_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        # 4) Rename columns into your "final_results" schema
        final_results = all_slides.rename(columns={
            "True_Label": "True_WSI_Label_Normal",
            "Soft_Pred":  "Predicted_WSI_Label_Weighted",
            # "Grad_Pred":  "Predicted_WSI_Label_Weighted_Densenet_Gradcam"
        })

        # 5) Save the detailed WSI‑level CSV
        wsi_csv_path = os.path.join(results_dir, "wsi_level_results.csv")
        final_results.to_csv(wsi_csv_path, index=False)
        f.write(f"\nSaved WSI-level results to: {wsi_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        fold_metrics = pd.DataFrame({
            "Fold": list(range(1, len(maj_accs)+1)),
            "Maj_Acc": maj_accs,
            "Maj_Sens": maj_sens,
            "Maj_Spec": maj_specs,
            "Maj_Prec": maj_precs,
            "Maj_F1": maj_f1s,

        })
        fold_metrics_path = os.path.join(results_dir, "wsi_level_per_testfold_metrics.csv")
        fold_metrics.to_csv(fold_metrics_path, index=False)
        f.write(f"\nSaved per-fold WSI-level metrics to: {fold_metrics_path}\n")
        f.flush(); os.fsync(f.fileno())

        f.write("\n=== Final Cross-Fold WSI-Level Metrics ===\n")
        print("\n=== Final Cross-Fold WSI-Level Metrics ===")

        # 6) Compute per‑method WSI metrics
        
        for name, vals in zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"],
            [maj_accs, maj_sens, maj_specs, maj_precs, maj_f1s]
        ):
            summarize(vals, name, "Majority Voting", f)

        for name, vals in zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"],
            [soft_accs, soft_sens, soft_specs, soft_precs, soft_f1s]
        ):
            summarize(vals, name, "Softmax Voting", f)

        # 7) Identify & save any mismatches across methods
        mismatches = final_results[
            (final_results["True_WSI_Label_Normal"] != final_results["Predicted_WSI_Label_Normal"]) |
            (final_results["True_WSI_Label_Normal"] != final_results["Predicted_WSI_Label_Weighted"]) ]
        mismatch_csv_path = os.path.join(results_dir, "wsi_level_mismatches.csv")
        mismatches.to_csv(mismatch_csv_path, index=False)
        f.write(f"\nSaved all‑method mismatches to: {mismatch_csv_path} (count: {len(mismatches)})\n")
        f.flush(); os.fsync(f.fileno())

        logger.info("End of Whole Slide Image Classification")
        f.write("\nEnd of Whole Slide Image Classification\n")
        f.flush(); os.fsync(f.fileno())
        
    
        ########################################################################################################################################
        logger.info(" Box-Plot Visualization for patch level classification of Whole Slide Image")
        # visualize box-plots = drawing the box around the location of the patches in each wsi image normal with green and benign weith red color
        visualize_patch_locations(args, all_patch_names, all_patch_labels, all_patch_predictions, all_wsi_names, all_patch_coordinates)
    
        logger.info(" End of Box-Plot Visualization for patch level classification of Whole Slide Image")



if __name__ == "__main__":
    main()
