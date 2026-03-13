# coding=utf-8
"""
Data utility functions for generating stratified folds and creating
PyTorch DataLoaders for training and evaluation.
"""
import random
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


def _get_stratified_groups(df_copy):
    """Helper to group WSI names by label and contrast."""
    contrasts = ['normal', 'dark']
    groups = {}
    for lbl in df_copy['Binary_Label'].unique():
        for ct_val in contrasts:
            mask = (df_copy['Binary_Label'] == lbl) & (df_copy['contrast_norm'] == ct_val)
            bucket = df_copy.loc[mask, 'WSI'].tolist()
            random.shuffle(bucket)
            groups[(int(lbl), ct_val)] = bucket
    return groups


def folds_generate(metadata_df, num_folds=5, seed=None):
    """
    Generate stratified folds based on Binary_Label and Contrast.

    Args:
        metadata_df: DataFrame containing WSI metadata.
        num_folds: Number of folds to create.
        seed: Random seed for shuffling.

    Returns:
        List of lists containing WSI names for each fold.
    """
    if seed is not None:
        random.seed(seed)

    df_copy = metadata_df.copy()
    df_copy['contrast_norm'] = df_copy['Contrast'].str.strip().str.lower()
    
    groups = _get_stratified_groups(df_copy)
    folds = [[] for _ in range(num_folds)]

    for samples in groups.values():
        total = len(samples)
        curr_idx = 0
        for i in range(num_folds):
            # Calculate size for this specific fold to handle remainders
            take = total // num_folds + (1 if i < total % num_folds else 0)
            folds[i].extend(samples[curr_idx: curr_idx + take])
            curr_idx += take
    return folds


def get_loader(args, trainset, valset, testset, all_trainin_validation_set):
    """
    Creates and returns DataLoaders for training, validation, testing, and tuning.
    """
    is_distributed = args.local_rank != -1
    
    # Common DataLoader configuration
    dl_kwargs = {
        "num_workers": 4,
        "pin_memory": True
    }

    train_sampler = DistributedSampler(trainset) if is_distributed else RandomSampler(trainset)
    
    train_loader = DataLoader(
        trainset, sampler=train_sampler, batch_size=args.train_batch_size, **dl_kwargs
    )
    val_loader = DataLoader(
        valset, sampler=SequentialSampler(valset), batch_size=args.eval_batch_size, **dl_kwargs
    )
    test_loader = DataLoader(
        testset, sampler=SequentialSampler(testset), batch_size=args.eval_batch_size, **dl_kwargs
    )
    tune_loader = DataLoader(
        all_trainin_validation_set, sampler=SequentialSampler(all_trainin_validation_set),
        batch_size=args.eval_batch_size, **dl_kwargs
    )
    
    return train_loader, val_loader, test_loader, tune_loader
