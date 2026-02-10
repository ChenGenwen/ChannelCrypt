"""
锁定模型
对训练好的confuse模型应用Q变换和块置换
生成发布版本的locked模型(未授权用户无法正常使用)
"""

import os
import sys
import yaml
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model, get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:  # 添加 encoding='utf-8'
        config = yaml.safe_load(f)
    return config


def lock_model(model, block_manager, key_manager, device='cpu'):
    """
    锁定模型
    
    流程:
    1. 对每个接口层的每个块应用Q变换 (块内混合)
    2. 对每个接口层应用块置换π (块间重排)
    
    Args:
        model: PyTorch模型
        block_manager: BlockManager实例
        key_manager: KeyManager实例
        device: 设备
    """
    print("\nApplying lock transformations...")
    
    model.eval()  # 设为评估模式
    
    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        c = info['c']
        g = info['g']
        
        print(f"\nProcessing layer: {layer_name}")
        print(f"  Block size: {c}, Num blocks: {g}")
        
        # ========================================
        # 步骤1: 应用Q变换到所有块
        # ========================================
        print("  [1/2] Applying Q transforms...")
        for i in range(g):
            # 生成Q矩阵
            Q = key_manager.generate_Q(layer_name, i, c, device=device)
            
            # 应用Q变换: W_new = Q @ Flat(W_old)
            block_manager.apply_Q_to_block(layer_name, i, Q)
            
            if (i + 1) % 4 == 0 or (i + 1) == g:
                print(f"    Blocks {i+1}/{g} completed")
        
        # ========================================
        # 步骤2: 应用块置换π
        # ========================================
        print("  [2/2] Applying block permutation...")
        
        # 生成选块集合S
        S = key_manager.generate_selection(layer_name, g, rho=0.5)
        
        # 生成置换π
        pi = key_manager.generate_permutation(layer_name, S, g)
        
        # 应用置换
        pi_tensor = torch.from_numpy(pi).long()
        block_manager.permute_blocks(layer_name, pi_tensor)
        
        print(f"    Permuted {len(S)} selected blocks")
        print(f"    {g - len(S)} blocks remain in place")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path =  r'D:\Model IP Protection\locked\GoogLeNet-cifar-10\config\config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("Locking Confuse Model")
    print("="*60)
    
    # ============================================================
    # 2. 设置设备
    # ============================================================
    device = torch.device('cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ============================================================
    # 3. 加载confuse模型
    # ============================================================
    confuse_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['confuse_model']
    )
    
    if not os.path.exists(confuse_model_path):
        raise FileNotFoundError(f"Confuse model not found: {confuse_model_path}")
    
    print(f"\nLoading confuse model from: {confuse_model_path}")
    model = load_model(
        confuse_model_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    print("Confuse model loaded successfully!")
    
    # ============================================================
    # 4. 初始化管理器
    # ============================================================
    print("\nInitializing managers...")
    
    # 获取接口配置
    interface_config = get_interface_config(config['model']['arch'])
    
    # 初始化BlockManager
    block_manager = BlockManager(model, interface_config)
    
    # 初始化KeyManager
    key_manager = KeyManager(config['confuse']['key'])
    
    # ============================================================
    # 5. 执行锁定操作
    # ============================================================
    with torch.no_grad():  # 锁定操作不需要梯度
        lock_model(model, block_manager, key_manager, device)
    
    # ============================================================
    # 6. 保存locked模型
    # ============================================================
    locked_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['locked_model']
    )
    
    print(f"\nSaving locked model to: {locked_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': config['model']['arch'],
        'num_classes': config['model']['num_classes'],
        'interface_config': interface_config,
    }, locked_model_path)
    
    print("\n" + "="*60)
    print("Model locked successfully!")
    print("="*60)
    print("\nNote: The locked model has degraded performance without the key.")
    print("Use assemble.py with the correct key to restore performance.")


if __name__ == '__main__':
    main()