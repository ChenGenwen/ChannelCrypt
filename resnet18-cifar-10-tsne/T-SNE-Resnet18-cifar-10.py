"""
ResNet18 + CIFAR-10 T-SNE 可视化（含 TEE-Assisted）
绘制 4 张图：Base / Locked / Assembled / TEE-Assisted
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ===========================
# 路径配置
# ===========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)  # ChannelCrypt/
PROJECT_DIR = os.path.join(BASE_DIR, 'resnet18-cifar-10')
sys.path.insert(0, PROJECT_DIR)

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
DATA_DIR = r"D:\Model IP Protection\data"
OUTPUT_DIR = SCRIPT_DIR

from models.interfaces import get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager

CONFUSE_KEY = "4a7d1ed414474e4033ac29ccb8653d9b"
RHO = 0.5

# ===========================
# 模型配置（按顺序绘制）
# ===========================
MODEL_ENTRIES = [
    ('base',      'base_resnet18_cifar10.pth'),
    ('locked',    'locked_resnet18_cifar10.pth'),
    ('assembled', 'assembled_resnet18_cifar10.pth'),
    ('tee',       'locked_resnet18_cifar10.pth'),   # locked + TEE hooks
]

# T-SNE 参数
N_SAMPLES = 5000
PERPLEXITY = 30
N_ITER = 1000
RANDOM_STATE = 42


# ===========================
# 模型定义
# ===========================
def create_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


# ===========================
# 数据加载
# ===========================
def load_cifar10_testset(data_dir, n_samples=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform
    )
    indices = []
    samples_per_class = n_samples // 10
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(testset):
        class_indices[label].append(idx)
    np.random.seed(RANDOM_STATE)
    for class_id in range(10):
        sampled = np.random.choice(class_indices[class_id], samples_per_class, replace=False)
        indices.extend(sampled)
    subset = Subset(testset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
    return loader, testset.classes


# ===========================
# 特征提取
# ===========================
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = model.avgpool.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.squeeze()
        if len(self.features.shape) == 1:
            self.features = self.features.unsqueeze(0)

    def __call__(self, x):
        _ = self.model(x)
        return self.features

    def remove(self):
        self.hook.remove()


def extract_features(model, dataloader, device):
    model.eval()
    extractor = FeatureExtractor(model)
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = extractor(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    extractor.remove()
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"  Extracted features shape: {features.shape}")
    return features, labels


# ===========================
# TEE Hook 设置（参考 tee_simulate.py）
# ===========================
def _fix_bn(bn, pi_inv, c, g):
    old = {k: getattr(bn, k).data.clone()
           for k in ('weight', 'bias', 'running_mean', 'running_var')}
    for j in range(g):
        src = pi_inv[j]
        d, s = slice(j * c, (j + 1) * c), slice(src * c, (src + 1) * c)
        for k in old:
            getattr(bn, k).data[d] = old[k][s]


def _make_hook(pi_inv, Q_T_list, c, g):
    def hook_fn(module, inp, output):
        B, C, H, W = output.shape
        corrected = torch.empty_like(output)
        for j in range(g):
            corrected[:, j * c:(j + 1) * c] = output[:, pi_inv[j] * c:(pi_inv[j] + 1) * c]
        for j in range(g):
            shard = corrected[:, j * c:(j + 1) * c].reshape(B, c, H * W)
            corrected[:, j * c:(j + 1) * c] = torch.matmul(
                Q_T_list[j], shard).reshape(B, c, H, W)
        return corrected
    return hook_fn


def setup_tee_hooks(model, block_manager, key_manager, rho, device):
    """在 locked 模型上挂 TEE 校正 hook，返回 hook 列表用于后续 remove"""
    hooks = []
    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        conv, bn = info['conv'], info['bn']
        c, g = info['c'], info['g']

        S = key_manager.generate_selection(layer_name, g, rho)
        pi = key_manager.generate_permutation(layer_name, S, g)
        pi_inv = key_manager.invert_permutation(pi)

        Q_T_list = []
        for i in range(g):
            Q = key_manager.generate_Q(layer_name, i, c, device=device)
            Q_T_list.append(Q.T.contiguous())

        _fix_bn(bn, pi_inv, c, g)
        hook = conv.register_forward_hook(_make_hook(pi_inv, Q_T_list, c, g))
        hooks.append(hook)
    return hooks


# ===========================
# 模型加载
# ===========================
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ===========================
# T-SNE
# ===========================
def compute_tsne(features, labels, perplexity=30, n_iter=1000):
    print(f"  Computing T-SNE (perplexity={perplexity}, max_iter={n_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=RANDOM_STATE, verbose=0)
    features_2d = tsne.fit_transform(features)
    print(f"  T-SNE completed\n")
    return features_2d


def plot_tsne(features_2d, labels, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for class_id in range(10):
        mask = labels == class_id
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[class_id]], label=class_names[class_id],
                    alpha=0.6, s=20, edgecolors='none')
    plt.legend(loc='lower left', ncol=2, fontsize=14, markerscale=2.5, framealpha=0.9)
    ax = plt.gca()
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ===========================
# 主函数
# ===========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 数据
    print("\n[1/3] Loading CIFAR-10 test set...")
    dataloader, class_names = load_cifar10_testset(DATA_DIR, n_samples=N_SAMPLES)
    print(f"  Sampled {N_SAMPLES} images")

    # 2. 逐模型提取特征 + T-SNE
    for model_name, model_file in MODEL_ENTRIES:
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_name.upper()}")
        print(f"{'=' * 60}")

        model_path = os.path.join(CHECKPOINT_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model not found: {model_path}")
            continue

        print(f"  Loading: {model_path}")
        model = create_resnet18_cifar10(num_classes=10).to(device)
        model = load_checkpoint(model, model_path, device)

        hooks = []
        if model_name == 'tee':
            # 初始化 BlockManager + KeyManager，挂 TEE hooks
            interface_config = get_interface_config('resnet18')
            bm = BlockManager(model, interface_config)
            km = KeyManager(CONFUSE_KEY)
            hooks = setup_tee_hooks(model, bm, km, RHO, device)
            print(f"  TEE hooks registered on {len(hooks)} layers")

        print(f"  Extracting features...")
        features, labels = extract_features(model, dataloader, device)

        # 清理
        for h in hooks:
            h.remove()
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"  Computing T-SNE...")
        features_2d = compute_tsne(features, labels, perplexity=PERPLEXITY, n_iter=N_ITER)

        save_path = os.path.join(OUTPUT_DIR, f"tsne_resnet18_cifar10_{model_name}.pdf")
        plot_tsne(features_2d, labels, class_names, model_name, save_path)

    print(f"\n{'=' * 60}")
    print(f"All done → {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
