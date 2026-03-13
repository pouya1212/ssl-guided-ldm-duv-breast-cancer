# coding=utf-8
"""
Data utility functions for generating stratified folds and creating
PyTorch DataLoaders for training and evaluation.
"""
import random
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


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
    # pylint: disable=too-many-locals
    df_copy = metadata_df.copy()
    df_copy['contrast_norm'] = df_copy['Contrast'].str.strip().str.lower()
    contrasts = ['normal', 'dark']
    groups = {}
    for lbl in df_copy['Binary_Label'].unique():
        for ct_val in contrasts:
            key = (int(lbl), ct_val)
            # Split long filtering line to satisfy C0301
            mask = (df_copy['Binary_Label'] == lbl) & (df_copy['contrast_norm'] == ct_val)
            bucket = df_copy.loc[mask, 'WSI'].tolist()
            random.shuffle(bucket)
            groups[key] = bucket

    folds = [[] for _ in range(num_folds)]
    for samples in groups.values():
        n_samples = len(samples)
        per_fold = [n_samples // num_folds + (1 if i < n_samples % num_folds else 0)
                    for i in range(num_folds)]
        curr_idx = 0
        for fold_idx, take in enumerate(per_fold):
            folds[fold_idx].extend(samples[curr_idx: curr_idx + take])
            curr_idx += take
    return folds


def get_loader(args, trainset, valset, testset, all_trainin_validation_set):
    """
    Creates and returns DataLoaders for training, validation, testing, and tuning.
    """
    if args.local_rank == -1:
        train_sampler = RandomSampler(trainset)
    else:
        train_sampler = DistributedSampler(trainset)

    # Reformatting DataLoaders to fit within 100 characters
    train_loader = DataLoader(
        trainset, sampler=train_sampler, batch_size=args.train_batch_size,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        valset, sampler=SequentialSampler(valset), batch_size=args.eval_batch_size,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, sampler=SequentialSampler(testset), batch_size=args.eval_batch_size,
        num_workers=4, pin_memory=True
    )
    tune_loader = DataLoader(
        all_trainin_validation_set, sampler=SequentialSampler(all_trainin_validation_set),
        batch_size=args.eval_batch_size, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, test_loader, tune_loader
