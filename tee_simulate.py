"""
TEE 辅助推理模拟 — 跨实验组运行器（修正版）
================================================
修正内容:
  1. 自动从checkpoint探测输入通道数，解决MNIST(1ch)/CIFAR(3ch)兼容
  2. 推理前预热DataLoader和CUDA，确保计时公平
  3. DATA_DIR_OVERRIDE 覆盖config中的数据路径
  4. 兼容不同实验组的 get_interface_config 签名
"""

import os
import sys
import csv
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime

# ============================================================
#  全局配置
# ============================================================
EXPERIMENT_GROUPS = [
    'GoogLeNet-MNIST',
    'GoogLeNet-cifar-10',
    'GoogLeNet-cifar-100',
    # 'resnet18-ImageNet',
    'resnet18-MNIST',
    'resnet18-cifar-10',
    'resnet18-cifar-100',
    'vgg16-BN-MNIST',
    'vgg16-BN-cifar-10',
    'vgg16-BN-cifar-100',
    'wideresnet50-2-cifar-10',
    'wideresnet50-2-cifar-100',
    'wideresnet50-2-MNIST',
]

DATA_DIR_OVERRIDE = '/seu_share2/home/huangjie/230258664/Model_IP_Protection/data'
WARMUP_BATCHES = 10
TIMING_BATCHES = 50
OUTPUT_CSV = 'tee_results.csv'


# ============================================================
#  工具函数
# ============================================================
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_input_channels(checkpoint_path, device):
    """从 checkpoint 的 conv1 权重形状探测输入通道数"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    for key in ('conv1.weight', 'features.0.weight'):
        if key in sd:
            return sd[key].shape[1]
    return 3


def load_model_safe(checkpoint_path, arch, num_classes, device,
                    load_model_fn, create_model_fn):
    """
    安全加载模型：先尝试原始 load_model，
    若因通道数不匹配失败则探测正确通道数后重建。
    """
    try:
        return load_model_fn(checkpoint_path, arch=arch,
                             num_classes=num_classes, device=device)
    except RuntimeError as e:
        if 'size mismatch' not in str(e) and 'channels' not in str(e):
            raise
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        in_ch = detect_input_channels(checkpoint_path, device)
        try:
            model = create_model_fn(arch, num_classes, pretrained=False,
                                    input_channels=in_ch)
        except TypeError:
            model = create_model_fn(arch, num_classes, pretrained=False)
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        print(f"  Fallback load: detected input_channels={in_ch}")
        return model


def get_test_loader(config, input_channels=3):
    """创建测试集 DataLoader，input_channels 控制 MNIST 的通道转换"""
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    data_dir = config['dataset']['data_dir']
    name = config['dataset'].get('name', 'cifar100')
    bs = config['dataset']['batch_size']
    nw = config['dataset']['num_workers']

    if name == 'cifar10':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=tf)

    elif name == 'cifar100':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=tf)

    elif name == 'mnist':
        if input_channels == 3:
            tf = transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32),
                transforms.ToTensor(), transforms.Normalize(mean, std),
            ])
        else:
            m = mean[:1] if len(mean) > 1 else mean
            s = std[:1] if len(std) > 1 else std
            tf = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(), transforms.Normalize(m, s),
            ])
        ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=tf)

    elif name == 'imagenet':
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'imagenet', 'val'), tf)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)


def warmup_loader_and_cuda(model, loader, device, n_batches=5):
    """
    预热 DataLoader 工作进程 + CUDA 内核编译，
    消除首次推理的冷启动偏差，确保后续计时公平。
    """
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            _ = model(x.to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ============================================================
#  TEE 校正 hook
# ============================================================
def setup_tee_hooks(model, block_manager, key_manager, rho, device):
    hooks = []
    layer_eta = {}

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

        C_in, k = conv.weight.shape[1], conv.weight.shape[2]
        eta = c / (C_in * k * k)
        layer_eta[layer_name] = {
            'eta': eta, 'C_out': info['num_channels'],
            'C_in': C_in, 'k': k, 'c': c, 'g': g,
        }

        hook = conv.register_forward_hook(_make_hook(pi_inv.copy(), Q_T_list, c, g))
        hooks.append(hook)

    return hooks, layer_eta


def _fix_bn(bn, pi_inv, c, g):
    old = {k: getattr(bn, k).data.clone()
           for k in ('weight', 'bias', 'running_mean', 'running_var')}
    for j in range(g):
        src = pi_inv[j]
        d, s = slice(j*c, (j+1)*c), slice(src*c, (src+1)*c)
        for k in old:
            getattr(bn, k).data[d] = old[k][s]


def _make_hook(pi_inv, Q_T_list, c, g):
    def hook_fn(module, inp, output):
        B, C, H, W = output.shape
        corrected = torch.empty_like(output)
        for j in range(g):
            corrected[:, j*c:(j+1)*c] = output[:, pi_inv[j]*c:(pi_inv[j]+1)*c]
        for j in range(g):
            shard = corrected[:, j*c:(j+1)*c].reshape(B, c, H*W)
            corrected[:, j*c:(j+1)*c] = torch.matmul(
                Q_T_list[j], shard).reshape(B, c, H, W)
        return corrected
    return hook_fn


# ============================================================
#  推理 + 计时
# ============================================================
def timed_inference(model, loader, device):
    model.eval()
    correct = total = 0
    batch_times = []
    batch_idx = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total += y.size(0)
            correct += out.argmax(1).eq(y).sum().item()

            batch_idx += 1
            if batch_idx > WARMUP_BATCHES:
                batch_times.append((t1 - t0) * 1000.0)

    acc = 100.0 * correct / total
    avg_ms = float(np.mean(batch_times[:TIMING_BATCHES])) if batch_times else 0.0
    return acc, avg_ms


# ============================================================
#  单实验组
# ============================================================
def run_group(group_dir, device, writer):
    abs_dir = os.path.abspath(group_dir)
    cfg_path = os.path.join(abs_dir, 'config', 'config.yaml')
    if not os.path.isfile(cfg_path):
        print(f"  [SKIP] config not found"); return

    config = load_config(cfg_path)
    if DATA_DIR_OVERRIDE:
        config['dataset']['data_dir'] = DATA_DIR_OVERRIDE

    ckpt_dir = os.path.join(abs_dir, config['paths']['checkpoint_dir'].lstrip('./'))
    base_path = os.path.join(ckpt_dir, config['paths']['base_model'])
    locked_path = os.path.join(ckpt_dir, config['paths']['locked_model'])

    if not os.path.isfile(base_path):
        print(f"  [SKIP] base model not found: {base_path}"); return
    if not os.path.isfile(locked_path):
        print(f"  [SKIP] locked model not found: {locked_path}"); return

    # 动态导入
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)
    for mod in list(sys.modules.keys()):
        if mod.startswith(('models', 'core', 'utils')):
            del sys.modules[mod]

    from models import load_model, create_model, get_interface_config
    from core.blocks import BlockManager
    from core.crypto import KeyManager

    rho = config['confuse']['rho']
    arch = config['model']['arch']
    nc = config['model']['num_classes']
    use_cuda = device.type == 'cuda'

    # 探测输入通道数（从base checkpoint）
    in_ch = detect_input_channels(base_path, device)
    print(f"  Detected input_channels={in_ch}")

    test_loader = get_test_loader(config, input_channels=in_ch)

    # ========== A. Base 模型 ==========
    print(f"  [1/2] Base model...")
    base_model = load_model_safe(base_path, arch, nc, device, load_model, create_model)

    # 预热（消除 DataLoader 工作进程启动 + CUDA 内核编译的冷启动偏差）
    warmup_loader_and_cuda(base_model, test_loader, device)

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    base_acc, base_ms = timed_inference(base_model, test_loader, device)
    base_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else 0.0
    print(f"         acc={base_acc:.2f}%  time={base_ms:.2f}ms/batch  mem={base_mem:.1f}MB")
    del base_model
    if use_cuda:
        torch.cuda.empty_cache()

    # ========== B. Locked + TEE ==========
    print(f"  [2/2] Locked + TEE...")
    locked_model = load_model_safe(locked_path, arch, nc, device, load_model, create_model)

    try:
        iface_cfg = get_interface_config(arch, model=locked_model)
    except TypeError:
        iface_cfg = get_interface_config(arch)

    bm = BlockManager(locked_model, iface_cfg)
    km = KeyManager(config['confuse']['key'])
    hooks, layer_eta = setup_tee_hooks(locked_model, bm, km, rho, device)

    # 预热
    warmup_loader_and_cuda(locked_model, test_loader, device)

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    tee_acc, tee_ms = timed_inference(locked_model, test_loader, device)
    tee_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else 0.0

    for h in hooks:
        h.remove()
    del locked_model
    if use_cuda:
        torch.cuda.empty_cache()

    overhead_ms = tee_ms - base_ms
    overhead_pct = 100.0 * overhead_ms / base_ms if base_ms > 0 else 0.0
    mem_overhead = tee_mem - base_mem

    print(f"         acc={tee_acc:.2f}%  time={tee_ms:.2f}ms/batch  mem={tee_mem:.1f}MB")
    print(f"         overhead: {overhead_ms:+.2f}ms ({overhead_pct:+.2f}%)  mem: {mem_overhead:+.1f}MB")

    eta_parts = [f"{ln}: {ei['eta']*100:.2f}%" for ln, ei in layer_eta.items()]
    eta_str = ' | '.join(eta_parts)
    print(f"         η: {eta_str}")

    # 写 CSV
    writer.writerow({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'group': os.path.basename(group_dir),
        'arch': arch, 'dataset': config['dataset'].get('name', ''),
        'num_classes': nc,
        'base_acc': f"{base_acc:.2f}", 'tee_acc': f"{tee_acc:.2f}",
        'base_ms': f"{base_ms:.2f}", 'tee_ms': f"{tee_ms:.2f}",
        'overhead_ms': f"{overhead_ms:.2f}", 'overhead_pct': f"{overhead_pct:.2f}",
        'base_mem_mb': f"{base_mem:.1f}", 'tee_mem_mb': f"{tee_mem:.1f}",
        'mem_overhead_mb': f"{mem_overhead:.1f}", 'eta_detail': eta_str,
    })

    if abs_dir in sys.path:
        sys.path.remove(abs_dir)


# ============================================================
#  主入口
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("  TEE-Assisted Inference Simulation")
    print("=" * 65)
    print(f"Device : {device}")
    print(f"Groups : {len(EXPERIMENT_GROUPS)}")
    print("=" * 65)

    csv_path = os.path.join(script_dir, OUTPUT_CSV)
    fields = [
        'timestamp', 'group', 'arch', 'dataset', 'num_classes',
        'base_acc', 'tee_acc',
        'base_ms', 'tee_ms', 'overhead_ms', 'overhead_pct',
        'base_mem_mb', 'tee_mem_mb', 'mem_overhead_mb', 'eta_detail',
    ]
    exists = os.path.isfile(csv_path)
    f = open(csv_path, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        w.writeheader()

    for name in EXPERIMENT_GROUPS:
        gdir = os.path.join(script_dir, name)
        print(f"\n{'─'*65}\n  Group: {name}\n{'─'*65}")
        if not os.path.isdir(gdir):
            print(f"  [SKIP] directory not found"); continue
        try:
            run_group(gdir, device, w)
            f.flush()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    f.close()
    print(f"\n{'='*65}\n  Done → {csv_path}\n{'='*65}")


if __name__ == '__main__':
    main()
