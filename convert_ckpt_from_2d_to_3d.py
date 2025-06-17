import torch
from collections import OrderedDict
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def convert_2d_resnet_weights_to_3d_checkpoint(
    input_2d_checkpoint_path: str,
    output_3d_checkpoint_path: str,
    video_trunk_prefix: str = 'feature_extractor_video.trunk.',
    temporal_kernel_size: int = 3
):
    if not os.path.exists(input_2d_checkpoint_path):
        logger.error(f"错误: 输入文件 '{input_2d_checkpoint_path}' 不存在。")
        return

    logger.info(f"正在加载原始 2D 检查点: {input_2d_checkpoint_path}")
    try:
        checkpoint = torch.load(input_2d_checkpoint_path, map_location=torch.device('cpu'))
        original_model_state_dict = checkpoint.get("model", checkpoint)
    except Exception as e:
        logger.error(f"加载checkpoint失败: {e}")
        return

    converted_model_state_dict = OrderedDict()
    conversion_count = 0

    logger.info(f"开始转换 2D ResNet 权重到 3D ResNet, 时间维度卷积核大小为3")
    for key, val in original_model_state_dict.items():
        if key.startswith(video_trunk_prefix):
            if val.dim() == 4 and 'weight' in key:
                if 'downsample.0.weight' in key:
                    converted_val = val.unsqueeze(2) 
                    converted_model_state_dict[key] = converted_val
                    conversion_count += 1
                else:
                    converted_val = val.unsqueeze(2).repeat(1, 1, temporal_kernel_size, 1, 1) / temporal_kernel_size
                    converted_model_state_dict[key] = converted_val
                    conversion_count += 1
            elif 'bn' in key and ('weight' in key or 'bias' in key or 'running_mean' in key or 'running_var' in key):
                converted_model_state_dict[key] = val
            elif 'prelu' in key:
                converted_model_state_dict[key] = val
            else:
                converted_model_state_dict[key] = val
        else:
            converted_model_state_dict[key] = val
    logger.info(f"转换完成。共转换了 {conversion_count} 个卷积权重。")

    if "model" in checkpoint:
        checkpoint["model"] = converted_model_state_dict
    else:
        checkpoint = converted_model_state_dict

    try:
        torch.save(checkpoint, output_3d_checkpoint_path)
        logger.info("保存成功！")
    except Exception as e:
        logger.error(f"保存checkpoint失败: {e}")

if __name__ == '__main__':
    
    INPUT_2D_CHECKPOINT = "final_project_ckpt/pretrained_model.pth" 
    OUTPUT_3D_CHECKPOINT = "final_project_ckpt/pretrained_model_3d_pro.pth"
    VIDEO_TRUNK_PREFIX = 'feature_extractor_video.trunk.' 
    TARGET_TEMPORAL_KERNEL_SIZE = 3

    convert_2d_resnet_weights_to_3d_checkpoint(
        INPUT_2D_CHECKPOINT,
        OUTPUT_3D_CHECKPOINT,
        video_trunk_prefix=VIDEO_TRUNK_PREFIX,
        temporal_kernel_size=TARGET_TEMPORAL_KERNEL_SIZE
    )
    
