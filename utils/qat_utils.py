"""
Quantization-Aware Training (QAT) Utilities
"""

import copy
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional


class QuantizableWrapper(nn.Module):
    """
    Wrapper to make any model quantizable for QAT
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def prepare_model_for_qat(
    model: nn.Module,
    qconfig_backend: str = 'fbgemm'
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training
    
    Args:
        model: FP32 model
        qconfig_backend: 'fbgemm' for x86, 'qnnpack' for ARM
    
    Returns:
        Model prepared for QAT
    """
    # Deep copy to avoid modifying original
    model_qat = copy.deepcopy(model)
    model_qat.train()
    
    # Wrap model
    model_wrapped = QuantizableWrapper(model_qat)
    
    # Set QAT qconfig
    if qconfig_backend == 'fbgemm':
        qconfig = quant.get_default_qat_qconfig('fbgemm')
    elif qconfig_backend == 'qnnpack':
        qconfig = quant.get_default_qat_qconfig('qnnpack')
    else:
        qconfig = quant.get_default_qat_qconfig('fbgemm')
    
    model_wrapped.qconfig = qconfig
    
    # Prepare for QAT
    # This inserts fake quantization modules
    model_prepared = quant.prepare_qat(model_wrapped, inplace=False)
    
    return model_prepared


def convert_qat_to_quantized(model_qat: nn.Module) -> nn.Module:
    """
    Convert QAT model to fully quantized INT8 model
    
    Args:
        model_qat: Model trained with QAT
    
    Returns:
        Quantized INT8 model
    """
    model_qat.eval()
    
    # Convert fake quantization to real quantization
    model_quantized = quant.convert(model_qat, inplace=False)
    
    return model_quantized


def freeze_batchnorm(model: nn.Module) -> None:
    """
    Freeze batch normalization layers during QAT
    This can improve stability in some cases
    
    Args:
        model: Model with batch norm layers
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            # Freeze parameters
            for param in module.parameters():
                param.requires_grad = False


def enable_observer(model: nn.Module, enabled: bool = True) -> None:
    """
    Enable or disable observers in QAT model
    Observers track statistics during training
    
    Args:
        model: QAT model
        enabled: Whether to enable observers
    """
    for module in model.modules():
        if hasattr(module, 'observer'):
            if enabled:
                module.observer.enable_observer()
            else:
                module.observer.disable_observer()


def load_fp32_weights_into_qat(
    qat_model: nn.Module,
    fp32_checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load FP32 pretrained weights into QAT model
    
    Args:
        qat_model: QAT-prepared model
        fp32_checkpoint_path: Path to FP32 checkpoint
        device: Device to load on
    
    Returns:
        QAT model with loaded weights
    """
    # Load FP32 checkpoint
    checkpoint = torch.load(fp32_checkpoint_path, map_location=device)
    fp32_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Get QAT model's state dict
    qat_state_dict = qat_model.state_dict()
    
    # Map FP32 weights to QAT model
    # QAT model has additional keys (observer, scale, zero_point)
    # We only load the actual weights
    loaded_keys = []
    for key in fp32_state_dict.keys():
        # Try to find matching key in QAT model
        # QAT adds prefixes like 'model.' due to wrapper
        qat_key = 'model.' + key
        
        if qat_key in qat_state_dict:
            qat_state_dict[qat_key] = fp32_state_dict[key]
            loaded_keys.append(key)
        elif key in qat_state_dict:
            qat_state_dict[key] = fp32_state_dict[key]
            loaded_keys.append(key)
    
    # Load the updated state dict
    qat_model.load_state_dict(qat_state_dict, strict=False)
    
    print(f"Loaded {len(loaded_keys)} FP32 weight keys into QAT model")
    
    return qat_model


def get_qat_lr_schedule(
    base_lr: float,
    qat_epochs: int,
    schedule_type: str = 'cosine'
):
    """
    Get learning rate schedule for QAT
    QAT typically uses lower learning rate than initial training
    
    Args:
        base_lr: Base learning rate (typically 10x smaller than FP32 training)
        qat_epochs: Number of QAT epochs
        schedule_type: 'cosine', 'step', or 'constant'
    
    Returns:
        Function that returns LR for given epoch
    """
    if schedule_type == 'cosine':
        import math
        def lr_schedule(epoch):
            return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / qat_epochs))
        return lr_schedule
    
    elif schedule_type == 'step':
        def lr_schedule(epoch):
            if epoch < qat_epochs // 2:
                return base_lr
            elif epoch < 3 * qat_epochs // 4:
                return base_lr * 0.1
            else:
                return base_lr * 0.01
        return lr_schedule
    
    else:  # constant
        def lr_schedule(epoch):
            return base_lr
        return lr_schedule


def print_qat_statistics(model: nn.Module) -> None:
    """
    Print statistics about fake quantization observers
    Useful for debugging QAT
    
    Args:
        model: QAT model
    """
    print("\n" + "="*70)
    print("QAT Observer Statistics")
    print("="*70)
    
    observer_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'observer'):
            observer_count += 1
            if hasattr(module.observer, 'min_val'):
                min_val = module.observer.min_val.item() if hasattr(module.observer.min_val, 'item') else None
                max_val = module.observer.max_val.item() if hasattr(module.observer.max_val, 'item') else None
                if min_val is not None and max_val is not None:
                    print(f"{name}: min={min_val:.4f}, max={max_val:.4f}")
    
    print(f"\nTotal observers: {observer_count}")
    print("="*70 + "\n")


def compare_fp32_qat_int8(
    fp32_metrics: dict,
    qat_fp32_metrics: dict,
    int8_metrics: dict,
    model_name: str
) -> None:
    """
    Print comparison between FP32, QAT (before conversion), and INT8
    
    Args:
        fp32_metrics: Original FP32 metrics
        qat_fp32_metrics: QAT model metrics (before INT8 conversion)
        int8_metrics: Final INT8 metrics
        model_name: Model name for display
    """
    print("\n" + "="*80)
    print(f"QAT Results Comparison - {model_name}")
    print("="*80)
    
    metrics_to_compare = ['accuracy', 'balanced_accuracy', 'f1_macro']
    
    print(f"{'Metric':<25} {'FP32 Original':<15} {'QAT (FP32)':<15} {'QAT INT8':<15} {'Improvement':<15}")
    print("-"*80)
    
    for metric in metrics_to_compare:
        fp32_val = fp32_metrics.get(metric, 0.0)
        qat_fp32_val = qat_fp32_metrics.get(metric, 0.0)
        int8_val = int8_metrics.get(metric, 0.0)
        
        improvement = int8_val - fp32_val
        improvement_pct = (improvement / fp32_val * 100) if fp32_val > 0 else 0.0
        
        print(f"{metric:<25} {fp32_val:<15.4f} {qat_fp32_val:<15.4f} {int8_val:<15.4f} {improvement_pct:>+7.2f}%")
    
    print("="*80)
    
    # Degradation analysis
    fp32_acc = fp32_metrics.get('accuracy', 0.0)
    int8_acc = int8_metrics.get('accuracy', 0.0)
    degradation = (1.0 - int8_acc / fp32_acc) * 100 if fp32_acc > 0 else 0.0
    
    print(f"\nAccuracy degradation: {degradation:.2f}%")
    
    if degradation < 1.0:
        print("✅ Excellent! <1% degradation")
    elif degradation < 3.0:
        print("✅ Good! <3% degradation")
    elif degradation < 5.0:
        print("⚠️  Acceptable. 3-5% degradation")
    else:
        print("❌ High degradation! >5%")
    
    print("="*80 + "\n")
