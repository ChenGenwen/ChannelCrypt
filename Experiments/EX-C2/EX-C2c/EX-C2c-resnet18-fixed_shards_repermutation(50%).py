"""
Data Ratio Attack Experiment - ResNet18 (CIFAR-10 / CIFAR-100)

Attack scenario:
  - Permutation recovery is fixed at 50% (floor(0.5 * |S_t|) blocks restored)
  - Q transforms are NOT recovered (no key)
  - Attacker has varying fractions of training data: 10%, 20%, 30%
  - For each fraction, remaining train data + official test set = evaluation test set

Experiment:
  For train_fraction in {10%, 20%, 30%}:
    - Partially restore 50% of permuted blocks (fixed)
    - Fine-tune all parameters for 100 epochs with fixed lr=1e-6
    - Test schedule: every epoch for epochs 1-15, then every 3 epochs

Output CSV (e.g. resnet18-cifar10.csv):
  Columns: epoch, Acc(10%), Acc(20%), Acc(30%)
  Written after each fraction completes (columns added left to right).
"""

import os
import sys
import csv
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from datetime import datetime

# ============================================================
# Project root (absolute path)
# ============================================================
PROJECT_ROOT = r"D:\Model IP Protection\locked\resnet18-cifar-10"
sys.path.insert(0, PROJECT_ROOT)

from models.interfaces import get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager

# ============================================================
# Dataset selection (change only this line)
# ============================================================
DATASET = 'cifar10'   # 'cifar10' | 'cifar100'

# ============================================================
# Dataset config table
# ============================================================
DATASET_CONFIG = {
    'cifar10': {
        'num_classes':    10,
        'input_channels': 3,
        'mean':           (0.4914, 0.4822, 0.4465),
        'std':            (0.2023, 0.1994, 0.2010),
        'loader':         torchvision.datasets.CIFAR10,
    },
    'cifar100': {
        'num_classes':    100,
        'input_channels': 3,
        'mean':           (0.5071, 0.4867, 0.4408),
        'std':            (0.2675, 0.2565, 0.2761),
        'loader':         torchvision.datasets.CIFAR100,
    },
}

# ============================================================
# Paths
# ============================================================
ARCH     = 'resnet18'
DATA_DIR = r"D:\Model IP Protection\data"

LOCKED_MODEL_MAP = {
    'cifar10':  r"D:\Model IP Protection\locked\locked_resnet18_cifar10.pth",
    'cifar100': r"D:\Model IP Protection\locked\locked_resnet18_cifar100.pth",
}
LOCKED_MODEL = LOCKED_MODEL_MAP[DATASET]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, ARCH + '-' + DATASET + '-data_ratio.csv')

# ============================================================
# Key
# ============================================================
CONFUSE_KEY = "4a7d1ed414474e4033ac29ccb8653d9b"

# ============================================================
# Hyperparameters
# ============================================================
TRAIN_FRACTIONS  = [0.4, 0.5, 0.6]   # attacker's data fractions
FIXED_RECOVERY   = 0.5               # fixed permutation recovery ratio
RHO              = 0.5               # must match lock.py

FINETUNE_EPOCHS  = 100
TEST_BOUNDARY    = 15
TEST_INTERVAL    = 3
FINETUNE_LR      = 1e-6
FINETUNE_BATCH   = 64
WEIGHT_DECAY     = 1e-4
MOMENTUM         = 0.9

# ============================================================
# Device & seed
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42


# ============================================================
# Model helpers
# ============================================================
def _adapt_resnet_for_small_images(model, input_channels=3):
    model.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=64,
        kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def create_model(arch, num_classes, input_channels=3):
    import torchvision.models as tv_models
    if arch == 'resnet18':
        model = tv_models.resnet18(weights=None)
        model = _adapt_resnet_for_small_images(model, input_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError('Unsupported arch: ' + arch)
    return model


def load_model(checkpoint_path, arch, num_classes, input_channels, device):
    model = create_model(arch, num_classes, input_channels)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ============================================================
# Helpers
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def should_test(epoch):
    return (epoch <= TEST_BOUNDARY
            or epoch % TEST_INTERVAL == 0
            or epoch == FINETUNE_EPOCHS)


def build_eval_epochs():
    epochs = {0}
    for e in range(1, FINETUNE_EPOCHS + 1):
        if should_test(e):
            epochs.add(e)
    return sorted(epochs)


def col_name(fraction):
    return 'Acc(' + str(int(fraction * 100)) + '%)'


def write_csv(curves, eval_epochs):
    """Overwrite OUTPUT_CSV with all completed fraction columns."""
    done_fracs = [f for f in TRAIN_FRACTIONS if f in curves]
    fieldnames = ['epoch'] + [col_name(f) for f in done_fracs]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in eval_epochs:
            row = {'epoch': epoch}
            for frac in done_fracs:
                acc = curves[frac].get(epoch, '')
                row[col_name(frac)] = '{:.4f}'.format(acc) if acc != '' else ''
            writer.writerow(row)


# ============================================================
# Data loading
# ============================================================
def get_data_loaders(dataset_name, data_dir, train_fraction, batch_size):
    """
    Train set : train_fraction of full training data (with augmentation)
    Test  set : remaining (1-train_fraction) training data (no aug)
                + official test split (no aug) -- merged
    """
    cfg          = DATASET_CONFIG[dataset_name]
    mean, std    = cfg['mean'], cfg['std']
    DatasetClass = cfg['loader']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_aug   = DatasetClass(root=data_dir, train=True,
                                    download=True, transform=transform_train)
    full_train_plain = DatasetClass(root=data_dir, train=True,
                                    download=True, transform=transform_test)

    total      = len(full_train_aug)
    num_attack = int(math.floor(train_fraction * total))

    rng             = np.random.default_rng(SEED)
    perm            = rng.permutation(total)
    attack_indices  = perm[:num_attack].tolist()
    holdout_indices = perm[num_attack:].tolist()

    attack_trainset  = Subset(full_train_aug,   attack_indices)
    holdout_set      = Subset(full_train_plain, holdout_indices)
    official_testset = DatasetClass(root=data_dir, train=False,
                                    download=True, transform=transform_test)
    combined_testset = ConcatDataset([holdout_set, official_testset])

    train_loader = DataLoader(attack_trainset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(combined_testset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# Partial permutation restore (fixed at FIXED_RECOVERY)
# ============================================================
def restore_permutation(model, block_manager, key_manager, recovery_ratio, rho):
    """
    Restore floor(recovery_ratio * |S_t|) block positions per layer.
    Q transforms are NOT restored.
    """
    print('\n  [Restore] recovery_ratio = ' + str(int(recovery_ratio * 100)) + '% (fixed)')

    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        g    = info['g']

        S      = key_manager.generate_selection(layer_name, g, rho=rho)
        pi     = key_manager.generate_permutation(layer_name, S, g)
        pi_inv = key_manager.invert_permutation(pi)

        m = len(S)
        k = int(math.floor(recovery_ratio * m))

        if k == 0:
            print('    ' + layer_name + ': recover 0/' + str(m) + ' blocks (no change)')
            continue

        rng_local    = np.random.default_rng(
            SEED + abs(hash(layer_name + str(recovery_ratio))) % (2 ** 31)
        )
        S_array      = np.array(S)
        chosen_local = rng_local.choice(m, size=k, replace=False)
        recover_set  = set(S_array[chosen_local].tolist())

        partial_pi = np.arange(g, dtype=np.int64)
        for j in recover_set:
            partial_pi[j] = pi_inv[j]

        block_manager.permute_blocks(
            layer_name, torch.from_numpy(partial_pi).long()
        )
        print('    ' + layer_name + ': recover ' + str(k) + '/' + str(m) +
              ' blocks (indices ' + str(sorted(recover_set)) + ')')


# ============================================================
# Training / evaluation
# ============================================================
def finetune_epoch(model, train_loader, criterion, optimizer,
                   device, epoch, num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(
        train_loader,
        desc='    Epoch [{:3d}/{}] train'.format(epoch, num_epochs),
        leave=False,
        dynamic_ncols=True
    )
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred  = outputs.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': '{:.4f}'.format(loss.item()),
            'acc':  '{:.2f}%'.format(100.0 * correct / total),
        })

    return total_loss / len(train_loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(test_loader, desc='    Evaluating',
                leave=False, dynamic_ncols=True)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        _, pred  = outputs.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()
        pbar.set_postfix({'acc': '{:.2f}%'.format(100.0 * correct / total)})

    return total_loss / len(test_loader), 100.0 * correct / total


# ============================================================
# Single fraction experiment
# ============================================================
def run_one_fraction(train_fraction):
    """
    Run the full fine-tuning experiment for one train_fraction.
    Returns dict {epoch: test_acc} for all evaluated epochs.
    """
    set_seed(SEED)

    cfg   = DATASET_CONFIG[DATASET]
    model = load_model(
        LOCKED_MODEL,
        arch=ARCH,
        num_classes=cfg['num_classes'],
        input_channels=cfg['input_channels'],
        device=DEVICE
    )

    interface_config = get_interface_config(ARCH)
    block_manager    = BlockManager(model, interface_config)
    key_manager      = KeyManager(CONFUSE_KEY)

    with torch.no_grad():
        restore_permutation(
            model, block_manager, key_manager,
            recovery_ratio=FIXED_RECOVERY,
            rho=RHO
        )

    # Build data loaders specific to this fraction
    train_loader, test_loader = get_data_loaders(
        DATASET, DATA_DIR, train_fraction, FINETUNE_BATCH
    )
    print('  Train samples: ' + str(len(train_loader.dataset))
          + '  |  Test samples: ' + str(len(test_loader.dataset)))

    criterion = nn.CrossEntropyLoss()
    _, acc0   = evaluate(model, test_loader, criterion, DEVICE)
    print('    Before finetune: test_acc = {:.2f}%'.format(acc0))

    acc_log  = {0: acc0}
    best_acc = acc0

    optimizer = optim.SGD(
        model.parameters(),
        lr=FINETUNE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    epoch_pbar = tqdm(
        range(1, FINETUNE_EPOCHS + 1),
        desc='  frac=' + str(int(train_fraction * 100)) + '%',
        dynamic_ncols=True
    )

    for epoch in epoch_pbar:
        _, train_acc = finetune_epoch(
            model, train_loader, criterion, optimizer,
            DEVICE, epoch, FINETUNE_EPOCHS
        )

        if should_test(epoch):
            _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            best_acc    = max(best_acc, test_acc)
            acc_log[epoch] = test_acc
            epoch_pbar.set_postfix({
                'train': '{:.2f}%'.format(train_acc),
                'test':  '{:.2f}%'.format(test_acc),
                'best':  '{:.2f}%'.format(best_acc),
            })
        else:
            epoch_pbar.set_postfix({'train': '{:.2f}%'.format(train_acc)})

    final_acc = acc_log[FINETUNE_EPOCHS]
    print('\n  -- frac=' + str(int(train_fraction * 100)) + '% done'
          + ' | final=' + '{:.2f}'.format(final_acc) + '%'
          + ', best=' + '{:.2f}'.format(best_acc) + '% --')
    return acc_log


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    SEP = '=' * 60
    print(SEP)
    print('Data Ratio Attack Experiment')
    print('  Dataset          : ' + DATASET)
    print('  Arch             : ' + ARCH)
    print('  Train fractions  : ' + str([str(int(f * 100)) + '%' for f in TRAIN_FRACTIONS]))
    print('  Fixed recovery   : ' + str(int(FIXED_RECOVERY * 100)) + '% of permuted blocks')
    print('  Finetune epochs  : ' + str(FINETUNE_EPOCHS)
          + '  (every epoch until ' + str(TEST_BOUNDARY)
          + ', then every ' + str(TEST_INTERVAL) + ')')
    print('  LR               : ' + str(FINETUNE_LR) + ' (fixed)')
    print('  Device           : ' + str(DEVICE))
    print('  Output CSV       : ' + OUTPUT_CSV)
    print(SEP)

    if not os.path.exists(LOCKED_MODEL):
        raise FileNotFoundError(
            'Locked model not found: ' + LOCKED_MODEL + '\n'
            'Please run lock.py first.'
        )

    eval_epochs = build_eval_epochs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curves = {}
    for frac in TRAIN_FRACTIONS:
        print('\n' + SEP)
        print('  Train fraction = ' + str(int(frac * 100)) + '%')
        print(SEP)

        curves[frac] = run_one_fraction(frac)

        write_csv(curves, eval_epochs)
        print('  CSV updated (' + str(len(curves)) + ' col(s)): ' + OUTPUT_CSV)

    # Summary
    print('\n' + SEP)
    print('Summary  (best / final  test_acc%)')
    print(SEP)
    print('  {:>8}  {:>8}  {:>8}'.format('Frac', 'Best', 'Final'))
    print('  ' + '-' * 32)
    for frac in TRAIN_FRACTIONS:
        log   = curves[frac]
        best  = max(log.values())
        final = log[FINETUNE_EPOCHS]
        print('  {:>7}%  {:>8.2f}  {:>8.2f}'.format(int(frac * 100), best, final))
    print(SEP)
    print('\nCSV saved : ' + OUTPUT_CSV)
    print('Finished  : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()