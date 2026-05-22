#!/usr/bin/env python3
"""
AID-VAR图像质量评估脚本 - 基于torch_fidelity
==============================================

使用torch_fidelity库计算指定目录中生成图像的FID和IS分数。
支持与真实数据集的对比评估，提供完整的图像质量分析。

主要功能：
- 计算FID (Fréchet Inception Distance) 分数
- 计算IS (Inception Score) 分数
- 支持多种数据集路径自动检测
- 提供详细的评估报告和可视化

作者: AID-VAR Team
修改日期: 2025-01-20
"""

import os
import sys
import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aid_fid_is_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入torch_fidelity用于FID/IS计算
try:
    from torch_fidelity import calculate_metrics
    HAS_TORCH_FIDELITY = True
    logger.info("✅ torch_fidelity可用，将使用官方标准FID/IS计算")
except ImportError:
    HAS_TORCH_FIDELITY = False
    logger.error("❌ torch_fidelity不可用，请安装: pip install torch-fidelity")
    raise ImportError("需要安装torch_fidelity库: pip install torch-fidelity")

class DirectoryFIDISEvaluator:
    """
    目录FID/IS评估器
    
    专门用于计算指定目录中图像的FID和IS分数
    """
    
    def __init__(self, 
                 generated_dir: str,
                 real_dir: Optional[str] = None,
                 output_dir: str = 'fid_is_evaluation_results',
                 device: str = 'auto'):
        """
        初始化评估器
        
        Args:
            generated_dir: 生成图像目录路径
            real_dir: 真实图像目录路径（用于FID计算）
            output_dir: 结果输出目录
            device: 计算设备 (auto/cuda/cpu)
        """
        self.generated_dir = Path(generated_dir)
        self.real_dir = Path(real_dir) if real_dir else None
        self.output_dir = Path(output_dir)
        
        # 设置计算设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        # 验证输入目录
        self._validate_directories()
        
        # 设置随机种子以确保可重现性
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f"🔧 目录FID/IS评估器初始化完成")
        logger.info(f"   生成图像目录: {self.generated_dir}")
        logger.info(f"   真实图像目录: {self.real_dir or '未指定(仅计算IS)'}")
        logger.info(f"   输出目录: {self.output_dir}")
        logger.info(f"   计算设备: {self.device}")
    
    def _validate_directories(self):
        """验证输入目录的有效性"""
        # 检查生成图像目录
        if not self.generated_dir.exists():
            raise ValueError(f"生成图像目录不存在: {self.generated_dir}")
        
        # 统计生成图像数量
        generated_images = self._count_images(self.generated_dir)
        if generated_images == 0:
            raise ValueError(f"生成图像目录中没有找到图像文件: {self.generated_dir}")
        
        logger.info(f"   生成图像数量: {generated_images}")
        
        # 检查真实图像目录（如果指定）
        if self.real_dir:
            if not self.real_dir.exists():
                logger.warning(f"真实图像目录不存在: {self.real_dir}")
                self.real_dir = None
            else:
                real_images = self._count_images(self.real_dir)
                if real_images == 0:
                    logger.warning(f"真实图像目录中没有找到图像文件: {self.real_dir}")
                    self.real_dir = None
                else:
                    logger.info(f"   真实图像数量: {real_images}")
    
    def _count_images(self, directory: Path) -> int:
        """统计目录中的图像文件数量"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        count = 0
        
        for ext in image_extensions:
            count += len(list(directory.glob(f'*{ext}')))
            count += len(list(directory.glob(f'*{ext.upper()}')))
        
        return count
    
    def _find_imagenet_dataset(self) -> Optional[Path]:
        """
        自动查找ImageNet数据集路径
        
        Returns:
            ImageNet验证集路径，如果找不到则返回None
        """
        # 常见的ImageNet数据集路径
        potential_paths = [
            # 项目中常见的路径模式
            "/home/intern/zkx/Dataset/imagenet-1k/datasets--ILSVRC--imagenet-1k/snapshots/4603483700ee984ea9debe3ddbfdeae86f6489eb/data/val",
            "/home/intern/zkx/Dataset/imagenet-1k/val", 
            "/home/data/imagenet/val",
            "/data/imagenet/val",
            "/dataset/imagenet/val",
            "/datasets/imagenet/val",
            "/data/ImageNet/val",
            "/dataset/ImageNet/val",
            # HuggingFace格式
            "/home/intern/zkx/Dataset/imagenet-1k/datasets--ILSVRC--imagenet-1k/snapshots/*/data/val",
            # 其他常见位置
            "./imagenet/val",
            "../imagenet/val",
            "../datasets/imagenet/val",
        ]
        
        for path_pattern in potential_paths:
            if '*' in path_pattern:
                # 处理通配符路径
                import glob
                matching_paths = glob.glob(path_pattern)
                for path in matching_paths:
                    path_obj = Path(path)
                    if path_obj.exists() and self._count_images(path_obj) > 1000:
                        logger.info(f"✅ 自动找到ImageNet验证集: {path_obj}")
                        return path_obj
            else:
                path_obj = Path(path_pattern)
                if path_obj.exists() and self._count_images(path_obj) > 1000:
                    logger.info(f"✅ 自动找到ImageNet验证集: {path_obj}")
                    return path_obj
        
        logger.warning("⚠️ 未能自动找到ImageNet数据集")
        return None
    
    def calculate_is_score(self) -> Dict[str, float]:
        """
        计算IS (Inception Score) 分数
        
        Returns:
            包含IS均值和标准差的字典
        """
        logger.info("📊 开始计算IS分数...")
        
        try:
            # 使用torch_fidelity计算IS
            metrics = calculate_metrics(
                input1=str(self.generated_dir),
                cuda=self.device == 'cuda',
                isc=True,  # 计算Inception Score
                verbose=True
            )
            
            is_results = {
                'inception_score_mean': float(metrics['inception_score_mean']),
                'inception_score_std': float(metrics['inception_score_std'])
            }
            
            logger.info(f"✅ IS计算完成 - 均值: {is_results['inception_score_mean']:.4f}, "
                       f"标准差: {is_results['inception_score_std']:.4f}")
            
            return is_results
            
        except Exception as e:
            logger.error(f"❌ IS计算失败: {str(e)}")
            raise
    
    def calculate_fid_score(self, real_dir: Optional[Path] = None) -> Optional[float]:
        """
        计算FID (Fréchet Inception Distance) 分数
        
        Args:
            real_dir: 真实图像目录，如果不指定则使用初始化时的目录
            
        Returns:
            FID分数，如果计算失败或无真实数据则返回None
        """
        # 确定真实图像目录
        target_real_dir = real_dir or self.real_dir
        
        if not target_real_dir:
            logger.warning("⚠️ 未指定真实图像目录，跳过FID计算")
            return None
        
        if not target_real_dir.exists():
            logger.warning(f"⚠️ 真实图像目录不存在，跳过FID计算: {target_real_dir}")
            return None
        
        logger.info(f"📊 开始计算FID分数...")
        logger.info(f"   生成图像: {self.generated_dir}")
        logger.info(f"   真实图像: {target_real_dir}")
        
        try:
            # 使用torch_fidelity计算FID，启用递归搜索
            metrics = calculate_metrics(
                input1=str(self.generated_dir),
                input2=str(target_real_dir),
                cuda=self.device == 'cuda',
                fid=True,  # 计算FID
                verbose=True,
                samples_find_deep=True  # 启用递归搜索子目录
            )
            
            fid_score = float(metrics['frechet_inception_distance'])
            
            logger.info(f"✅ FID计算完成 - 分数: {fid_score:.4f}")
            
            return fid_score
            
        except Exception as e:
            logger.error(f"❌ FID计算失败: {str(e)}")
            raise
    
    def calculate_all_metrics(self, auto_find_imagenet: bool = True) -> Dict:
        """
        计算所有可用的评估指标
        
        Args:
            auto_find_imagenet: 是否自动查找ImageNet数据集
            
        Returns:
            包含所有计算指标的字典
        """
        logger.info("🚀 开始计算FID和IS指标...")
        
        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'generated_dir': str(self.generated_dir),
                'real_dir': str(self.real_dir) if self.real_dir else None,
                'device': self.device,
                'generated_image_count': self._count_images(self.generated_dir)
            },
            'metrics': {}
        }
        
        # 计算IS分数
        try:
            is_results = self.calculate_is_score()
            results['metrics']['inception_score'] = is_results
        except Exception as e:
            logger.error(f"IS计算失败: {e}")
            results['metrics']['inception_score'] = {'error': str(e)}
        
        # 计算FID分数
        fid_real_dir = self.real_dir
        
        # 如果没有指定真实数据集且允许自动查找，尝试找到ImageNet
        if not fid_real_dir and auto_find_imagenet:
            fid_real_dir = self._find_imagenet_dataset()
            if fid_real_dir:
                results['evaluation_info']['auto_found_real_dir'] = str(fid_real_dir)
        
        if fid_real_dir:
            try:
                fid_score = self.calculate_fid_score(fid_real_dir)
                if fid_score is not None:
                    results['metrics']['frechet_inception_distance'] = fid_score
                    results['evaluation_info']['real_image_count'] = self._count_images(fid_real_dir)
            except Exception as e:
                logger.error(f"FID计算失败: {e}")
                results['metrics']['frechet_inception_distance'] = {'error': str(e)}
        else:
            logger.info("📝 未找到真实图像数据，仅计算IS分数")
            results['metrics']['frechet_inception_distance'] = None
        
        return results
    
    def create_sample_visualization(self, num_samples: int = 16):
        """
        创建生成样本的可视化网格
        
        Args:
            num_samples: 要显示的样本数量
        """
        logger.info(f"🖼️ 创建样本可视化 ({num_samples}张)...")
        
        try:
            # 收集图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(self.generated_dir.glob(f'*{ext}')))
                image_files.extend(list(self.generated_dir.glob(f'*{ext.upper()}')))
            
            if len(image_files) == 0:
                logger.warning("未找到图像文件用于可视化")
                return
            
            # 随机选择样本
            np.random.shuffle(image_files)
            selected_files = image_files[:min(num_samples, len(image_files))]
            
            # 计算网格尺寸
            grid_size = int(np.ceil(np.sqrt(len(selected_files))))
            
            # 创建图像网格
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
            if grid_size == 1:
                axes = [[axes]]
            elif grid_size > 1 and len(axes.shape) == 1:
                axes = [axes]
            
            for i, ax_row in enumerate(axes):
                for j, ax in enumerate(ax_row):
                    idx = i * grid_size + j
                    if idx < len(selected_files):
                        # 加载并显示图像
                        try:
                            img = Image.open(selected_files[idx]).convert('RGB')
                            ax.imshow(img)
                            ax.set_title(f'Sample {idx+1}', fontsize=8)
                        except Exception as e:
                            logger.warning(f"加载图像失败 {selected_files[idx]}: {e}")
                            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                    
                    ax.axis('off')
            
            plt.tight_layout()
            
            # 保存可视化
            vis_path = self.output_dir / 'sample_visualization.png'
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 样本可视化保存至: {vis_path}")
            
        except Exception as e:
            logger.error(f"创建可视化失败: {e}")
    
    def print_results(self, results: Dict):
        """
        打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print("\n" + "="*80)
        print("🏆 AID-VAR 图像质量评估结果")
        print("="*80)
        
        # 基本信息
        info = results['evaluation_info']
        print(f"\n📊 评估信息:")
        print(f"   生成图像目录: {info['generated_dir']}")
        print(f"   生成图像数量: {info['generated_image_count']}")
        print(f"   计算设备: {info['device']}")
        print(f"   评估时间: {info['timestamp']}")
        
        if info.get('real_dir'):
            print(f"   真实图像目录: {info['real_dir']}")
            if info.get('real_image_count'):
                print(f"   真实图像数量: {info['real_image_count']}")
        
        if info.get('auto_found_real_dir'):
            print(f"   自动找到真实数据: {info['auto_found_real_dir']}")
        
        # 评估指标
        metrics = results['metrics']
        print(f"\n🎯 评估指标:")
        
        # IS分数
        if 'inception_score' in metrics:
            is_data = metrics['inception_score']
            if 'error' in is_data:
                print(f"   IS计算失败: {is_data['error']}")
            else:
                is_mean = is_data['inception_score_mean']
                is_std = is_data['inception_score_std']
                print(f"   Inception Score: {is_mean:.4f} ± {is_std:.4f}")
                
                # IS分数解释
                print(f"   IS分数解释:")
                print(f"     - 分数越高表示图像质量越好，多样性越丰富")
                print(f"     - 典型范围: 1.0-50.0")
                print(f"     - 真实ImageNet约为11.2")
                if is_mean >= 10.0:
                    print(f"     - ✅ 生成质量优秀")
                elif is_mean >= 5.0:
                    print(f"     - ⚠️ 生成质量中等")
                else:
                    print(f"     - ❌ 生成质量较低")
        
        # FID分数
        if 'frechet_inception_distance' in metrics:
            fid_data = metrics['frechet_inception_distance']
            if fid_data is None:
                print(f"   FID: 未计算 (无真实数据)")
            elif isinstance(fid_data, dict) and 'error' in fid_data:
                print(f"   FID计算失败: {fid_data['error']}")
            else:
                fid_score = fid_data
                print(f"   FID Score: {fid_score:.4f}")
                
                # FID分数解释
                print(f"   FID分数解释:")
                print(f"     - 分数越低表示生成图像与真实图像越相似")
                print(f"     - 完美匹配的FID为0")
                print(f"     - 典型范围: 5-100+")
                if fid_score <= 10.0:
                    print(f"     - ✅ 生成质量极佳")
                elif fid_score <= 30.0:
                    print(f"     - ✅ 生成质量优秀") 
                elif fid_score <= 50.0:
                    print(f"     - ⚠️ 生成质量中等")
                else:
                    print(f"     - ❌ 生成质量较低")
        
        print("="*80)
    
    def save_results(self, results: Dict):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果字典
        """
        # 保存详细结果
        results_file = self.output_dir / 'fid_is_evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建简化摘要
        summary = {
            'directory': str(self.generated_dir),
            'image_count': results['evaluation_info']['generated_image_count'],
            'timestamp': results['evaluation_info']['timestamp'],
        }
        
        metrics = results['metrics']
        if 'inception_score' in metrics and 'error' not in metrics['inception_score']:
            summary['IS_mean'] = metrics['inception_score']['inception_score_mean']
            summary['IS_std'] = metrics['inception_score']['inception_score_std']
        
        if 'frechet_inception_distance' in metrics and metrics['frechet_inception_distance'] is not None:
            if not isinstance(metrics['frechet_inception_distance'], dict):
                summary['FID'] = metrics['frechet_inception_distance']
        
        summary_file = self.output_dir / 'evaluation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 创建文本报告
        report_file = self.output_dir / 'evaluation_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("AID-VAR 图像质量评估报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"评估目录: {self.generated_dir}\n")
            f.write(f"图像数量: {results['evaluation_info']['generated_image_count']}\n")
            f.write(f"评估时间: {results['evaluation_info']['timestamp']}\n\n")
            
            if 'inception_score' in metrics and 'error' not in metrics['inception_score']:
                is_data = metrics['inception_score']
                f.write(f"Inception Score: {is_data['inception_score_mean']:.4f} ± {is_data['inception_score_std']:.4f}\n")
            
            if 'frechet_inception_distance' in metrics and metrics['frechet_inception_distance'] is not None:
                if not isinstance(metrics['frechet_inception_distance'], dict):
                    f.write(f"FID Score: {metrics['frechet_inception_distance']:.4f}\n")
        
        logger.info(f"💾 评估结果已保存:")
        logger.info(f"   详细结果: {results_file}")
        logger.info(f"   评估摘要: {summary_file}")
        logger.info(f"   文本报告: {report_file}")
    
    def run_evaluation(self, auto_find_imagenet: bool = True, create_visualization: bool = True):
        """
        运行完整的评估流程
        
        Args:
            auto_find_imagenet: 是否自动查找ImageNet数据集进行FID计算
            create_visualization: 是否创建样本可视化
        """
        logger.info("🚀 开始运行AID-VAR图像质量评估")
        
        try:
        # 计算评估指标
            results = self.calculate_all_metrics(auto_find_imagenet=auto_find_imagenet)
        
            # 创建可视化
            if create_visualization:
                self.create_sample_visualization()
        
        # 打印结果
            self.print_results(results)
        
        # 保存结果
            self.save_results(results)
        
            logger.info("✅ 评估完成！")
            return results
        
        except Exception as e:
            logger.error(f"❌ 评估失败: {e}")
            raise

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description='AID-VAR图像质量评估 - 使用torch_fidelity计算FID和IS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 只计算IS分数
  python run_aid_var_evaluation.py --generated_dir /path/to/generated/images
  
  # 计算FID和IS分数
  python run_aid_var_evaluation.py --generated_dir /path/to/generated --real_dir /path/to/real
  
  # 使用默认目录
  python run_aid_var_evaluation.py
        """
    )
    
    parser.add_argument(
        '--generated_dir', 
        type=str, 
        default='/home/intern/Ligong/VAR/aid_fid_samples_d16_20250902_233415',
        help='生成图像目录路径'
    )
    parser.add_argument(
        '--real_dir', 
        type=str, 
        default=None,
        help='真实图像目录路径（用于FID计算）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='fid_is_evaluation_results',
        help='结果输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='计算设备'
    )
    parser.add_argument(
        '--no_auto_imagenet',
        action='store_true',
        help='禁用自动查找ImageNet数据集'
    )
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='禁用样本可视化'
    )
    
    args = parser.parse_args()
    
    print("🎯 AID-VAR 图像质量评估工具")
    print("="*60)
    print(f"基于torch_fidelity的标准FID/IS计算")
    print("="*60)
    
    try:
        # 创建评估器
        evaluator = DirectoryFIDISEvaluator(
            generated_dir=args.generated_dir,
            real_dir=args.real_dir,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # 运行评估
        results = evaluator.run_evaluation(
            auto_find_imagenet=not args.no_auto_imagenet,
            create_visualization=not args.no_visualization
        )
        
        print(f"\n📁 完整结果已保存至: {evaluator.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 