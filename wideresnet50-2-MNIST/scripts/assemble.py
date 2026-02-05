"""
恢复模型
对locked模型应用逆变换,恢复原始性能
只有持有正确密钥的用户才能成功恢复
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


def assemble_model(model, block_manager, key_manager, rho, device='cpu'):
    """
    恢复模型
    
    逆变换顺序(与lock相反):
    1. 逆置换π^{-1} (先恢复块的位置)
    2. 逆Q变换 Q^T (再恢复块内的混合)
    
    Args:
        model: PyTorch模型(locked版本)
        block_manager: BlockManager实例
        key_manager: KeyManager实例
        rho: 选块比例，必须与lock时一致（取自 config['confuse']['rho']）
        device: 设备
    """
    print("\nApplying assemble transformations...")
    
    model.eval()
    
    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        c = info['c']
        g = info['g']
        
        print(f"\nProcessing layer: {layer_name}")
        print(f"  Block size: {c}, Num blocks: {g}")
        
        # ========================================
        # 步骤1: 应用逆置换π^{-1}
        # ========================================
        print("  [1/2] Applying inverse permutation...")
        
        # 生成选块集合S(与lock时相同)
        S = key_manager.generate_selection(layer_name, g, rho=rho)
        
        # 生成置换π
        pi = key_manager.generate_permutation(layer_name, S, g)
        
        # 计算逆置换π^{-1}
        pi_inv = key_manager.invert_permutation(pi)
        
        # 应用逆置换
        pi_inv_tensor = torch.from_numpy(pi_inv).long()
        block_manager.permute_blocks(layer_name, pi_inv_tensor)
        
        print(f"    Inverse permutation applied")
        
        # ========================================
        # 步骤2: 应用逆Q变换(Q^T)到所有块
        # ========================================
        print("  [2/2] Applying inverse Q transforms...")
        for i in range(g):
            # 生成Q矩阵(与lock时相同)
            Q = key_manager.generate_Q(layer_name, i, c, device=device)
            
            # 应用逆Q变换: W_recovered = Q^T @ Flat(W_locked)
            Q_inv = Q.T  # 正交矩阵的逆等于其转置
            block_manager.apply_Q_to_block(layer_name, i, Q_inv)
            
            if (i + 1) % 4 == 0 or (i + 1) == g:
                print(f"    Blocks {i+1}/{g} completed")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("Assembling Locked Model")
    print("="*60)
    
    # ============================================================
    # 2. 设置设备
    # ============================================================
    device = torch.device('cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ============================================================
    # 3. 加载locked模型
    # ============================================================
    locked_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['locked_model']
    )
    
    if not os.path.exists(locked_model_path):
        raise FileNotFoundError(f"Locked model not found: {locked_model_path}")
    
    print(f"\nLoading locked model from: {locked_model_path}")
    model = load_model(
        locked_model_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    print("Locked model loaded successfully!")
    
    # ============================================================
    # 4. 初始化管理器
    # ============================================================
    print("\nInitializing managers...")
    
    # 获取接口配置
    interface_config = get_interface_config(config['model']['arch'], model=model)
    
    # 初始化BlockManager
    block_manager = BlockManager(model, interface_config)
    
    # 初始化KeyManager (使用相同的密钥)
    key_manager = KeyManager(config['confuse']['key'])
    
    print("\n⚠️  Make sure you are using the CORRECT key!")
    print(f"Current key: {config['confuse']['key'][:16]}...")
    
    # ============================================================
    # 5. 执行恢复操作
    # ============================================================
    with torch.no_grad():
        assemble_model(model, block_manager, key_manager,
                       rho=config['confuse']['rho'], device=device)
    
    # ============================================================
    # 6. 保存assembled模型
    # ============================================================
    assembled_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['assembled_model']
    )
    
    print(f"\nSaving assembled model to: {assembled_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': config['model']['arch'],
        'num_classes': config['model']['num_classes'],
    }, assembled_model_path)
    
    print("\n" + "="*60)
    print("Model assembled successfully!")
    print("="*60)
    print("\nYou can now test the assembled model with test.py")
    print("It should have similar performance to the confuse model.")


if __name__ == '__main__':
    main()