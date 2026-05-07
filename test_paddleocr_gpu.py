#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试PaddleOCR GPU初始化"""

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# 测试PaddlePaddle GPU
import paddle
print(f"PaddlePaddle版本: {paddle.__version__}")
print(f"CUDA编译: {paddle.device.is_compiled_with_cuda()}")
print(f"当前设备: {paddle.device.get_device()}")

# 测试PaddleOCR GPU初始化
from paddleocr import PaddleOCR
print("\n正在初始化PaddleOCR (GPU)...")

try:
    ocr = PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        lang='japan',
        device='gpu',  # 强制使用GPU
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=2.0,
    )
    print("✅ PaddleOCR GPU初始化成功！")
    
    # 测试简单的OCR识别（如果有测试图片）
    # import sys
    # if len(sys.argv) > 1:
    #     result = ocr.predict(sys.argv[1])
    #     print(f"识别结果: {result}")
    
except Exception as e:
    print(f"❌ PaddleOCR初始化失败: {e}")
    import traceback
    traceback.print_exc()
