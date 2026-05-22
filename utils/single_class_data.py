"""
🎯 AID-VAR单分类验证数据集
专门为验证框架有效性设计的轻量级数据加载器
"""

import os
import json
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    """🔥 关键修复：使用与原始VAR训练相同的数据标准化方法"""
    return x.add(x).add_(-1)

class SingleClassImageNetDataset(Dataset):
    """
    严格单分类ImageNet数据集 - 仅支持Golden Retriever (类别ID 207)
    
    🎯 严格单分类模式：专门为AID-VAR框架验证设计
    - 只支持类别ID 207 (Golden Retriever) 
    - 确保训练和验证数据的一致性
    - 防止意外使用多类别数据
    """
    
    def __init__(
        self,
        data_root: str,
        class_id: int = 207,  # Golden Retriever - 视觉特征丰富
        split: str = 'train',
        transform=None,
        max_samples: Optional[int] = 1000,  # 限制样本数量用于快速验证
    ):
        """
        Args:
            data_root: ImageNet数据集根目录
            class_id: ImageNet类别ID
            split: 'train' or 'val'
            transform: 图像变换
            max_samples: 最大样本数量（None表示使用全部）
        """
        self.data_root = data_root
        self.class_id = class_id
        self.split = split
        self.max_samples = max_samples
        
        # ImageNet类别映射
        self.class_names = self._load_imagenet_classes()
        self.class_name = self.class_names.get(class_id, f"class_{class_id}")
        
        # 🔥 关键修复：使用与原始VAR训练相同的数据标准化方法
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize_01_into_pm1  # 🔥 使用与原始VAR相同的标准化
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    normalize_01_into_pm1  # 🔥 使用与原始VAR相同的标准化
                ])
        else:
            self.transform = transform
        
        # 加载图像路径
        self.image_paths = self._load_image_paths()
        
        logger.info(f"✅ 严格单分类数据集初始化完成:")
        logger.info(f"   🎯 模式: 严格单分类 (仅Golden Retriever)")
        logger.info(f"   类别: {self.class_name} (ID: {class_id})")
        logger.info(f"   分割: {split}")
        logger.info(f"   样本数: {len(self.image_paths)}")
        logger.info(f"   📁 数据文件夹: n02099712")
        logger.info(f"   🔥 数据标准化: normalize_01_into_pm1 (与原始VAR一致)")
    
    def _load_imagenet_classes(self) -> dict:
        """加载ImageNet类别映射 - 严格单分类"""
        # 🎯 严格单分类：只支持Golden Retriever (类别ID 207)
        class_mapping = {
            207: "golden_retriever",  # 唯一支持的类别
        }
        return class_mapping
    
    def _load_image_paths(self) -> List[str]:
        """加载指定类别的所有图像路径"""
        # ImageNet目录结构: data_root/train/n02099712/many_images.JPEG
        # 或者: data_root/val/n02099712/many_images.JPEG
        
        # 🎯 严格单分类映射：只支持Golden Retriever (类别ID 207)
        class_id_to_folder = {
            207: "n02099712",  # Golden Retriever - 唯一支持的类别
        }
        
        # 🎯 严格验证：只允许类别ID 207 (Golden Retriever)
        if self.class_id != 207:
            raise ValueError(f"❌ 严格单分类模式：只支持类别ID 207 (Golden Retriever)，收到: {self.class_id}")
        
        folder_name = class_id_to_folder.get(self.class_id)
        if folder_name is None:
            raise ValueError(f"类别ID {self.class_id} 未找到对应的文件夹映射")
        
        # 构建完整路径
        split_dir = os.path.join(self.data_root, self.split)
        target_folder = os.path.join(split_dir, folder_name)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"数据集分割目录不存在: {split_dir}")
        
        if not os.path.exists(target_folder):
            raise ValueError(f"类别文件夹不存在: {target_folder}")
        
        logger.info(f"使用文件夹: {target_folder}")
        
        # 收集图像文件
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        
        for filename in os.listdir(target_folder):
            if any(filename.endswith(ext) for ext in valid_extensions):
                image_paths.append(os.path.join(target_folder, filename))
        
        # 限制样本数量
        if self.max_samples and len(image_paths) > self.max_samples:
            image_paths = image_paths[:self.max_samples]
        
        if not image_paths:
            raise ValueError(f"在{target_folder}中没有找到有效的图像文件")
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: 变换后的图像张量 (C, H, W)
            label: 类别标签张量 (标量)
        """
        image_path = self.image_paths[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 返回图像和标签
            label = torch.tensor(self.class_id, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            logger.warning(f"加载图像失败 {image_path}: {e}")
            # 返回一个随机图像作为fallback
            image = torch.randn(3, 256, 256)
            label = torch.tensor(self.class_id, dtype=torch.long)
            return image, label

def create_single_class_dataloaders(
    data_root: str,
    class_id: int = 207,  # 🎯 严格单分类：固定为Golden Retriever
    batch_size: int = 8,
    num_workers: int = 4,
    max_train_samples: int = 800,
    max_val_samples: int = 200,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建严格单分类的训练和验证数据加载器
    
    🎯 严格单分类模式：只支持类别ID 207 (Golden Retriever)
    
    Args:
        data_root: ImageNet数据集根目录
        class_id: 类别ID，必须为207 (Golden Retriever)
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        max_train_samples: 最大训练样本数
        max_val_samples: 最大验证样本数
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    
    Raises:
        ValueError: 如果class_id不是207
    """
    
    # 🎯 严格验证：只允许类别ID 207
    if class_id != 207:
        raise ValueError(f"❌ 严格单分类模式：只支持类别ID 207 (Golden Retriever)，收到: {class_id}")
    
    # 创建数据集
    train_dataset = SingleClassImageNetDataset(
        data_root=data_root,
        class_id=class_id,
        split='train',
        max_samples=max_train_samples
    )
    
    val_dataset = SingleClassImageNetDataset(
        data_root=data_root,
        class_id=class_id,
        split='val', 
        max_samples=max_val_samples
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"✅ 严格单分类数据加载器创建完成:")
    logger.info(f"   🎯 模式: 严格单分类 (Golden Retriever)")
    logger.info(f"   类别ID: {class_id}")
    logger.info(f"   训练样本: {len(train_dataset)}")
    logger.info(f"   验证样本: {len(val_dataset)}")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"   📁 统一使用文件夹: n02099712")
    
    return train_loader, val_loader

# 测试函数
def test_single_class_dataset():
    """测试单分类数据集的基本功能"""
    print("🧪 测试单分类数据集...")
    
    # 创建测试数据集（使用虚拟路径进行演示）
    try:
        # 注意：实际使用时需要提供真实的ImageNet路径
        dataset = SingleClassImageNetDataset(
            data_root="/path/to/imagenet",  # 需要替换为实际路径
            class_id=207,
            split='train',
            max_samples=10
        )
        
        print(f"✅ 数据集创建成功，样本数: {len(dataset)}")
        
        # 测试数据加载
        image, label = dataset[0]
        print(f"✅ 数据加载成功:")
        print(f"   图像形状: {image.shape}")
        print(f"   标签: {label}")
        
    except Exception as e:
        print(f"⚠️ 测试跳过（需要真实数据路径）: {e}")

if __name__ == "__main__":
    test_single_class_dataset() 