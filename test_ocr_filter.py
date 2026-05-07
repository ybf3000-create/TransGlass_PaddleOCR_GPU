#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：用鳄鱼图测试OCR置信度过滤效果
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEOCR_INIT_LOG'] = 'False'
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_use_mkldnn'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_IMAGE = r"D:\8AI\Claw\glass\R-C.jpg"
OUTPUT_DIR = os.path.join(os.path.dirname(TEST_IMAGE), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 手动测试不同置信度门槛 ==========
def run_test(conf_thresh):
    """用指定门槛测试"""
    from TransGlass_PaddleOCR_GPU import (
        ComicOCR, get_paddleocr_instance, ComicBubbleMerger
    )
    
    print(f"\n{'='*60}")
    print(f"测试：置信度门槛 = {conf_thresh}")
    print(f"{'='*60}")
    
    # 1. 先用 ComicOCR 获取原始识别结果（不合并）
    ocr = ComicOCR(lang='japan', merge_rule='rule2_moderate')
    ocr.merger = ComicBubbleMerger(rule_name='rule2_moderate', lang='japan')
    
    # 先跑一次OCR获取原始块（不合并）
    from PIL import Image
    image_path = TEST_IMAGE
    
    paddle_ocr = get_paddleocr_instance(lang='japan')
    result = paddle_ocr.ocr(image_path)
    
    original_blocks = []
    if result and len(result) > 0:
        ocr_result = result[0]
        rec_texts = ocr_result.get('rec_texts')
        rec_boxes = ocr_result.get('rec_boxes')
        rec_scores = ocr_result.get('rec_scores', [1.0]*len(rec_texts or []))
        
        if rec_texts and rec_boxes is not None:
            for i, (text, box) in enumerate(zip(rec_texts, rec_boxes)):
                if not text or not text.strip():
                    continue
                score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                box_flat = box.flatten() if hasattr(box, 'flatten') else box
                x_coords = box_flat[::2]
                y_coords = box_flat[1::2]
                original_blocks.append({
                    'x1': int(min(x_coords)),
                    'y1': int(min(y_coords)),
                    'x2': int(max(x_coords)),
                    'y2': int(max(y_coords)),
                    'text': text.strip(),
                    'confidence': score
                })
    
    print(f"\nPaddleOCR 总共识别到 {len(original_blocks)} 个文本块")
    
    # 2. 按置信度分两组
    kept = [b for b in original_blocks if b.get('confidence', 0) >= conf_thresh]
    filtered = [b for b in original_blocks if b.get('confidence', 0) < conf_thresh]
    
    print(f"\n置信度 ≥ {conf_thresh} (保留，参与合并): {len(kept)} 个")
    print(f"置信度 < {conf_thresh} (过滤掉): {len(filtered)} 个")
    
    print("\n--- 保留的块 ---")
    for b in kept:
        print(f"  [保留] 置信度:{b['confidence']:.3f} 文本:'{b['text']}'  "
              f"位置:({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
    
    print("\n--- 过滤掉的块 ---")
    for b in filtered:
        print(f"  [过滤] 置信度:{b['confidence']:.3f} 文本:'{b['text']}'  "
              f"位置:({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
    
    # 3. 合并保留的块
    if kept:
        merger = ComicBubbleMerger(rule_name='rule2_moderate', lang='japan')
        merged = merger.merge_text_blocks(kept)
        print(f"\n--- 合并后: {len(merged)} 个文本块 ---")
        for b in merged:
            print(f"  文字:'{b['text']}'  覆盖框:({b['bg_x1']},{b['bg_y1']})-({b['bg_x2']},{b['bg_y2']})  "
                  f"大小:{b['bg_x2']-b['bg_x1']}x{b['bg_y2']-b['bg_y1']}")
    else:
        merged = []
        print("\n--- 无可合并的块 ---")
    
    # 4. 生成可视化
    try:
        from PIL import Image, ImageDraw
        img = Image.open(TEST_IMAGE).convert('RGBA')
        overlay_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        
        # 绿色框：保留的原始块
        for b in kept:
            draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']], 
                          outline=(0, 200, 0, 255), width=2)
        
        # 红色框：过滤掉的
        for b in filtered:
            draw.rectangle([b['x1'], b['y1'], b['x2'], b['y2']], 
                          outline=(255, 0, 0, 180), width=1)
        
        # 蓝色半透明框：合并后的覆盖框
        for b in merged:
            draw.rectangle([b['bg_x1'], b['bg_y1'], b['bg_x2'], b['bg_y2']], 
                          fill=(0, 100, 255, 50), outline=(0, 100, 255, 255), width=3)
            draw.text((b['bg_x1'], max(0, b['bg_y1']-18)), f"'{b['text']}'", 
                     fill=(0, 100, 255, 255))
        
        result = Image.alpha_composite(img, overlay_img)
        out_path = os.path.join(OUTPUT_DIR, f"crocodile_conf_{conf_thresh:.2f}.png")
        result.save(out_path)
        print(f"\n可视化: {out_path}")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    return merged

if __name__ == "__main__":
    from PIL import Image, ImageDraw
    
    print("="*60)
    print("鳄鱼图 OCR 测试 - 不同置信度门槛对比")
    print("="*60)
    
    # 测试5种门槛值
    for thresh in [0.1, 0.2, 0.25, 0.3, 0.4]:
        run_test(thresh)
    
    print(f"\n{'='*60}")
    print(f"所有测试完成！查看 {OUTPUT_DIR} 目录对比可视化结果")
    print(f"{'='*60}")
