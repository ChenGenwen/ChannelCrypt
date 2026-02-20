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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
    'locked': 'locked_resnet18_cifar10.pth',
    'assembled': 'assembled_resnet18_cifar10.pth'
}

# T-SNE 参数
N_SAMPLES = 5000
PERPLEXITY = 30
N_ITER = 1000
RANDOM_STATE = 42

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
    
    # 均匀采样
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
# 修正后的 ResNet18 定义
# ===========================
def create_resnet18_cifar10(num_classes=10):
    """创建适配 CIFAR-10 的 ResNet18（直接返回模型，不包装）"""
    # 创建标准 ResNet18
    model = models.resnet18(weights=None)
    
    # CIFAR-10 适配：修改第一个卷积层
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 去掉 maxpool
    
    # 修改最后的全连接层
    model.fc = nn.Linear(512, num_classes)
    
    return model

# ===========================
# 特征提取器（Hook 方式）
# ===========================
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        # 在 avgpool 之后注册 hook
        self.hook = model.avgpool.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.squeeze()  # [B, 512]
        if len(self.features.shape) == 1:  # 处理 batch_size=1
            self.features = self.features.unsqueeze(0)
    
    def __call__(self, x):
        _ = self.model(x)
        return self.features
    
    def remove(self):
        self.hook.remove()

# ===========================
# 提取特征
# ===========================
def extract_features(model, dataloader, device):
    """提取模型的特征表示"""
    model.eval()
    extractor = FeatureExtractor(model)
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = extractor(images)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    extractor.remove()
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"Extracted features shape: {features.shape}")
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
    plt.figure(figsize=(10, 8))
    
    # 10 种颜色（CIFAR-10）
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for class_id in range(10):
        mask = labels == class_id
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_id]],
            label=class_names[class_id],
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    plt.legend(
    loc='lower left',   # 固定在左下角
    ncol=2,              # 双列
    fontsize=14,
    markerscale=2.5,
    framealpha=0.9
)
    
  
    # ===== 添加黑色边框 =====
    ax = plt.gca()
    
    # 显示所有边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(2)  # 可以调整粗细：1, 1.5, 2 等
    
    # 隐藏刻度标签但保留边框
    ax.set_xticks([])
    ax.set_yticks([])
    # ========================
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ===========================
# 主函数
# ===========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载数据集
    print("\n[1/4] Loading CIFAR-10 test set...")
    dataloader, class_names = load_cifar10_testset(DATA_DIR, n_samples=N_SAMPLES)
    print(f"Sampled {N_SAMPLES} images from test set")
    
    # 2. 对每个模型进行 T-SNE 可视化
    for model_name, model_file in MODEL_FILES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # 加载模型
        model_path = os.path.join(CHECKPOINT_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            print(f"Skipping {model_name}...")
            continue
        
        print(f"[2/4] Loading model from: {model_path}")
        
        # 创建模型（注意：这里不再使用 ResNet18 类，直接调用函数）
        model = create_resnet18_cifar10(num_classes=10).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # 智能加载 state_dict
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
        print("Model loaded successfully!")
        
        # 提取特征
        print(f"[3/4] Extracting features...")
        features, labels = extract_features(model, dataloader, device)
        
        # T-SNE 降维
        print(f"[4/4] Computing T-SNE...")
        features_2d = compute_tsne(features, labels, perplexity=PERPLEXITY, n_iter=N_ITER)
        
        # 绘制并保存
        title = f"T-SNE Visualization: {model_name.capitalize()} Model (ResNet18/CIFAR-10)"
        save_path = os.path.join(OUTPUT_DIR, f"tsne_resnet18_cifar10_{model_name}.pdf")
        plot_tsne(features_2d, labels, class_names, title, save_path)
    
    print(f"\n{'='*60}")
    print("All visualizations completed!")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()