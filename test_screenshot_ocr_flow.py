#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试截图→OCR完整流程"""

import os
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont

# 设置环境变量
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# 添加CUDNN到PATH
cudnn_path = os.path.join(os.path.dirname(__file__), 'venv_gpu', 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
if os.path.exists(cudnn_path):
    os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')
    print(f"✅ 添加CUDNN路径: {cudnn_path}")

def test_ocr_recognition():
    """测试完整的OCR识别流程"""
    print("="*60)
    print("测试OCR识别流程")
    print("="*60)
    
    # 1. 创建测试图片（模拟截图）
    print("\n[步骤1] 创建测试图片...")
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    # 画一些文字（用矩形代替，因为可能没有中文字体）
    draw.rectangle([50, 50, 350, 150], outline='black', width=2)
    draw.text((100, 80), "Test OCR 123", fill='black')
    
    # 保存到临时文件
    temp_dir = tempfile.gettempdir()
    test_img_path = os.path.join(temp_dir, 'test_screenshot.png')
    img.save(test_img_path)
    print(f"✅ 测试图片已保存: {test_img_path}")
    
    # 2. 初始化PaddleOCR
    print("\n[步骤2] 初始化PaddleOCR (GPU)...")
    try:
        from paddleocr import PaddleOCR
        import paddle
        
        print(f"PaddlePaddle版本: {paddle.__version__}")
        print(f"CUDA编译: {paddle.device.is_compiled_with_cuda()}")
        print(f"当前设备: {paddle.device.get_device()}")
        
        ocr = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang='japan',  # 测试日语识别
            device='gpu',
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_det_unclip_ratio=2.0,
        )
        print("✅ PaddleOCR初始化成功（GPU模式）")
        
    except Exception as e:
        print(f"❌ PaddleOCR初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 执行OCR识别
    print("\n[步骤3] 执行OCR识别...")
    try:
        result = ocr.ocr(test_img_path)
        print(f"✅ OCR识别成功！")
        print(f"识别结果: {result}")
        return True
        
    except Exception as e:
        print(f"❌ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理临时文件
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
            print(f"\n🗑️  已清理临时文件: {test_img_path}")

if __name__ == '__main__':
    success = test_ocr_recognition()
    if success:
        print("\n" + "="*60)
        print("✅ 完整流程测试通过！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 流程测试失败！请检查错误信息。")
        print("="*60)
        sys.exit(1)
