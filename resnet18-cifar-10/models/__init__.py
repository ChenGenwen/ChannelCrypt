"""
T-SNE Visualization for ResNet18 on CIFAR-10
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ===========================
# 配置路径
# ===========================
SCRIPT_DIR = r"D:\Model IP Protection\locking model scheme"
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, r"resnet18-cifar-10\checkpoints")
DATA_DIR = r"D:\Model IP Protection\data"
OUTPUT_DIR = SCRIPT_DIR

# 模型文件名
MODEL_FILES = {
    'base': 'base_resnet18_cifar10.pth',
    'homogeneous': 'homogeneous_resnet18_cifar10.pth', 
    'locked': 'locked_resnet18_cifar10.pth',
    'assembled': 'assembled_resnet18_cifar10.pth'
}

# T-SNE 参数
N_SAMPLES = 2000
PERPLEXITY = 30
N_ITER = 1000
RANDOM_STATE = 42

# ===========================
# 模型创建函数（从您的 models/__init__.py 复制）
# ===========================
def _adapt_resnet_for_cifar(model):
    """将标准ResNet适配为CIFAR-10（32×32输入）"""
    # 1. 替换conv1: 7×7 stride=2 → 3×3 stride=1
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # 2. 去掉maxpool
    model.maxpool = nn.Identity()
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """创建模型"""
    if arch == 'resnet18':
        # 创建标准ResNet18
        model = models.resnet18(weights=None)
        
        # 适配CIFAR-10的32×32输入
        model = _adapt_resnet_for_cifar(model)
        
        # 修改最后的全连接层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    return model

# ===========================
# 特征提取器（Hook 方式）
# ===========================
class FeatureExtractor:
    """使用 Hook 提取 avgpool 后的特征"""
    def __init__(self, model):
        self.model = model
        self.features = None
        
        # 在 avgpool 之后注册 hook
        self.hook = model.avgpool.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # output shape: [B, 512, 1, 1]
        self.features = output.squeeze()  # [B, 512]
        if len(self.features.shape) == 1:  # 处理 batch_size=1 的情况
            self.features = self.features.unsqueeze(0)
    
    def __call__(self, x):
        _ = self.model(x)
        return self.features
    
    def remove(self):
        self.hook.remove()

# ===========================
# 加载 CIFAR-10 数据集
# ===========================
def load_cifar10_testset(data_dir, n_samples=2000):
    """加载 CIFAR-10 测试集并采样"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=False,
        transform=transform
    )
    
    # 均匀采样（每类采样 n_samples/10 张）
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
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=2)
    
    return loader, testset.classes

# ===========================
# 提取特征
# ===========================
def extract_features(model, dataloader, device):
    """提取模型的特征表示"""
    model.eval()
    extractor = FeatureExtractor(model)
    
    features_list = []
    labels_list = []
    
    print("  Extracting features from batches...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            features = extractor(images)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"    Processed {(batch_idx + 1) * images.size(0)} / {N_SAMPLES} images")
    
    extractor.remove()
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"  ✓ Extracted features shape: {features.shape}\n")
    return features, labels

# ===========================
# T-SNE 降维与可视化
# ===========================
def compute_tsne(features, labels, perplexity=30, n_iter=1000):
    """计算 T-SNE 降维"""
    print(f"  Computing T-SNE (perplexity={perplexity}, max_iter={n_iter})...")
    
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        max_iter=n_iter,  # ← 改成 max_iter
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    features_2d = tsne.fit_transform(features)
    print(f"  ✓ T-SNE completed\n")
    return features_2d

def plot_tsne(features_2d, labels, class_names, title, save_path):
    """绘制 T-SNE 散点图"""
    print(f"  Plotting and saving to: {os.path.basename(save_path)}")
    
    plt.figure(figsize=(12, 10))
    
    # 10 种颜色（CIFAR-10）
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for class_id in range(10):
        mask = labels == class_id
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_id]],
            label=class_names[class_id],
            alpha=0.7,
            s=25,
            edgecolors='none'
        )
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('T-SNE Dimension 1', fontsize=14)
    plt.ylabel('T-SNE Dimension 2', fontsize=14)
    plt.legend(
        loc='upper right', 
        fontsize=11, 
        markerscale=2, 
        framealpha=0.95,
        ncol=2
    )
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved successfully\n")
    plt.close()

# ===========================
# 主函数
# ===========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"T-SNE Visualization for ResNet18 on CIFAR-10")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # 1. 加载数据集
    print("[Step 1/4] Loading CIFAR-10 test set...")
    dataloader, class_names = load_cifar10_testset(DATA_DIR, n_samples=N_SAMPLES)
    print(f"✓ Loaded {N_SAMPLES} images (200 per class)\n")
    
    # 2. 对每个模型进行 T-SNE 可视化
    for idx, (model_name, model_file) in enumerate(MODEL_FILES.items(), 1):
        print(f"{'─'*70}")
        print(f"[Step 2/4 - Model {idx}/4] Processing: {model_name.upper()}")
        print(f"{'─'*70}")
        
        # 检查模型文件是否存在
        model_path = os.path.join(CHECKPOINT_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"⚠ WARNING: Model not found at {model_path}")
            print(f"  Skipping {model_name}...\n")
            continue
        
        print(f"  Model file: {model_file}")
        
        # 创建模型
        print("  Creating model structure...")
        model = create_model(arch='resnet18', num_classes=10, pretrained=False)
        model = model.to(device)
        
        # 加载 checkpoint
        print("  Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 智能加载 state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                test_acc = checkpoint.get('test_acc', 0)
                print(f"  ✓ Loaded from epoch {epoch}, test_acc={test_acc:.2f}%")
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print(f"  ✓ Loaded model weights")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"  ✓ Loaded state dict")
            else:
                # 假设整个 dict 就是 state_dict
                model.load_state_dict(checkpoint)
                print(f"  ✓ Loaded state dict (direct)")
        else:
            # 直接是 state_dict
            model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded state dict (direct)")
        
        model.eval()
        
        # 提取特征
        print("\n[Step 3/4] Extracting features...")
        features, labels = extract_features(model, dataloader, device)
        
        # T-SNE 降维
        print("[Step 4/4] Running T-SNE dimensionality reduction...")
        features_2d = compute_tsne(features, labels, perplexity=PERPLEXITY, n_iter=N_ITER)
        
        # 绘制并保存
        title = f"T-SNE Visualization: {model_name.capitalize()} Model\n(ResNet18 on CIFAR-10)"
        save_path = os.path.join(OUTPUT_DIR, f"tsne_resnet18_cifar10_{model_name}.png")
        plot_tsne(features_2d, labels, class_names, title, save_path)
    
    print(f"{'='*70}")
    print("✅ All visualizations completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()