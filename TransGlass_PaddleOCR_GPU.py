#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransGlass-PaddleOCR CPU版
漫画翻译器 - 透明覆盖层版本
功能：截图 -> PaddleOCR识别 -> Ollama翻译 -> 透明层覆盖显示

快捷键：可在托盘菜单「快捷键设置」中自定义

环境要求：
  - Python 3.8 - 3.12（推荐 3.11）
  - paddlepaddle + paddleocr
  - PySide6, mss, pynput, Pillow, numpy, requests
"""

import os
import sys
import re
import json
import time
import subprocess
import threading
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont, ImageStat
import statistics

import warnings
warnings.filterwarnings('ignore')

# 抑制 urllib3 警告
import urllib3
urllib3.disable_warnings()

# 禁用 PaddleOCR 模型源检查（必须在导入前设置）
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEOCR_INIT_LOG'] = 'False'  # 尝试禁用初始化日志

# GPU 显存优化：动态分配，避免一次性占满 6GB（GTX 1060）
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
# CPU 推理时禁用 MKLDNN，避免 PaddlePaddle 3.2 的 OneDNN bug
os.environ['FLAGS_use_mkldnn'] = '0'

import requests

# ====================== OCR: PaddleOCR ======================
from paddleocr import PaddleOCR

PADDLEOCR_VERSION = None  # '2.x' 或 '3.x'
_paddleocr_instance = None
_ocr_lock = threading.Lock()

# 支持的OCR识别语言配置（用户选择要识别的源语言）
SUPPORTED_LANGUAGES = {
    'japan': {'name': 'Japanese', 'label': '日语', 'target_lang': '中文'},
    'korean': {'name': 'Korean', 'label': '韩语', 'target_lang': '中文'},
    'en': {'name': 'English', 'label': '英语', 'target_lang': '中文'},
}

# 默认OCR识别语言
DEFAULT_OCR_LANG = "japan"  # 默认识别日语

# 合并规则预设
# 横向/纵向合并倍率：k_multiplier 控制横向(同一行)，g_multiplier 控制纵向(不同行)
# 0=不合并该方向，推荐范围 0.5~2.0
MERGE_RULES = {
    'rule1_strict': {
        'name': '严格合并',
        'label': '规则1: 严格',
        'description': '仅合并距离很近的文本块，保持原始分组',
        'k_multiplier': 0.5, 'g_multiplier': 0.8,
    },
    'rule2_moderate': {
        'name': '适度合并',
        'label': '规则2: 适度',
        'description': '适度扩大合并范围，能合并同一气泡内的多行文字',
        'k_multiplier': 0.6, 'g_multiplier': 1.0,
    },
    'rule3_loose': {
        'name': '宽松合并',
        'label': '规则3: 宽松',
        'description': '较大范围合并，适合气泡间距较大的漫画',
        'k_multiplier': 0.8, 'g_multiplier': 1.5,
    },
    'rule4_none': {
        'name': '不合并',
        'label': '规则4: 不合并',
        'description': '不进行任何合并，每个识别到的文本块独立显示',
        'k_multiplier': 0.0, 'g_multiplier': 0.0,
    },
}

DEFAULT_MERGE_RULE = 'rule2_moderate'


def get_paddleocr_instance(lang: str = None):
    """获取 PaddleOCR 实例（懒加载，线程安全）
    
    Args:
        lang: 语言代码，支持 'en', 'japan', 'korean', 'ch' 等
    """
    global _paddleocr_instance, PADDLEOCR_VERSION
    with _ocr_lock:
        # 如果语言变化，需要重新初始化
        if _paddleocr_instance is not None and lang is not None:
            # 检查当前实例的语言是否匹配
            current_lang = getattr(_paddleocr_instance, 'lang', None)
            if current_lang != lang:
                print(f"[信息] 切换语言从 {current_lang} 到 {lang}，重新初始化PaddleOCR...")
                _paddleocr_instance = None
        
        if _paddleocr_instance is None:
            lang = lang or 'en'  # 默认英语
            # 检测 GPU 状态
            _gpu_available = False
            _gpu_device = "cpu"
            try:
                import paddle
                _gpu_available = paddle.device.is_compiled_with_cuda()
                _gpu_device = paddle.device.get_device() if _gpu_available else "cpu"
            except Exception:
                pass
            print(f"[信息] 正在初始化PaddleOCR (语言: {SUPPORTED_LANGUAGES.get(lang, {}).get('label', lang)}, 设备: {_gpu_device})...")
            try:
                # PaddleOCR 3.x 风格
                _paddleocr_instance = PaddleOCR(
                    use_textline_orientation=True,  # 启用方向分类，支持竖排文字
                    use_doc_orientation_classify=False,  # 关闭文档方向分类（避免预处理改变像素位置）
                    use_doc_unwarping=False,              # 关闭文档矫正（避免坐标偏移）
                    lang=lang,
                    device='gpu' if _gpu_available else 'cpu',  # 使用GPU（如果可用）
                    text_det_thresh=ocr_det_thresh,              # 检测阈值，越低检出越多
                    text_det_box_thresh=ocr_det_box_thresh,      # 文本框置信度阈值
                    text_det_unclip_ratio=2.0,    # 检测框外扩系数，越大框越松
                )
                PADDLEOCR_VERSION = '3.x'
                # 设置 use_dilation 和 score_mode（PaddleOCR 构造函数未暴露的参数）
                try:
                    post_op = _paddleocr_instance.paddlex_pipeline.text_det_model.post_op
                    post_op.use_dilation = True   # 膨胀分割图，检出小文字
                    post_op.score_mode = 'slow'   # 精确评分模式，边缘文字不易被过滤
                    print(f"[优化] 已启用 use_dilation=True, score_mode=slow")
                except AttributeError:
                    pass  # 某些版本可能不暴露 post_op
            except TypeError as e:
                if 'show_log' in str(e) or 'use_textline_orientation' in str(e):
                    # 回退到 PaddleOCR 2.x 风格
                    _paddleocr_instance = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False
                    )
                    PADDLEOCR_VERSION = '2.x'
                else:
                    raise
            # 保存语言设置
            _paddleocr_instance.lang = lang
            print(f"[成功] PaddleOCR {PADDLEOCR_VERSION} 初始化完成 (语言: {lang}, 设备: {_gpu_device})")
        return _paddleocr_instance


def reset_paddleocr_instance():
    """重置 PaddleOCR 实例（用于切换语言）"""
    global _paddleocr_instance
    with _ocr_lock:
        _paddleocr_instance = None
        print("[信息] PaddleOCR 实例已重置")


class ComicOCR:
    """漫画OCR识别器（PaddleOCR 引擎）"""

    def __init__(self, lang: str = None, merge_rule: str = None):
        self.lang = lang or 'en'  # 默认英语
        self.merge_rule = merge_rule or DEFAULT_MERGE_RULE
        self.merger = ComicBubbleMerger(rule_name=merge_rule, lang=self.lang)

    def _preprocess_tophat(self, image_path: str) -> Optional[str]:
        """使用 Top-hat 形态学预处理增强深色背景上文字检测
        对暗色调场景（游戏/动漫）中的亮色文字特别有效。
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            预处理后的图片路径，失败返回 None
        """
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            # 反转：文字变白底，背景变黑
            tophat_inv = cv2.bitwise_not(tophat)
            
            temp_path = image_path + ".tophat.jpg"
            cv2.imwrite(temp_path, tophat_inv)
            print(f"[预处理] Top-hat 形态学增强已应用")
            return temp_path
        except Exception as e:
            print(f"[预处理警告] Top-hat 增强失败: {e}")
            return None

    def recognize(self, image_path: str, merge_blocks: bool = True) -> tuple:
        """识别图片中的文字
        
        Args:
            image_path: 图片路径
            merge_blocks: 是否合并同一气泡框内的文字块
        
        Returns:
            (merged_blocks, original_blocks): 合并后的块和原始识别块
        """
        ocr = get_paddleocr_instance(lang=self.lang)
        print(f"[信息] 正在识别图片: {image_path} (语言: {self.lang})")

        # ========== 四边扩展10像素，解决边缘文字识别不到的问题 ==========
        PADDING = 10
        temp_path = None
        tophat_temp_path = None

        try:
            # 如果是文件路径，四边扩展白色区域
            if image_path and os.path.exists(image_path):
                with Image.open(image_path) as img:
                    orig_width, orig_height = img.size
                    new_w = orig_width + 2 * PADDING
                    new_h = orig_height + 2 * PADDING
                    if img.mode == 'RGBA':
                        new_img = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))
                    else:
                        new_img = Image.new('RGB', (new_w, new_h), (255, 255, 255))
                    new_img.paste(img, (PADDING, PADDING))
                    temp_path = image_path + ".temp_padded.jpg"
                    new_img.save(temp_path, "JPEG", quality=95)
                    print(f"[优化] 图片四边扩展 {PADDING} 像素 ({orig_width}x{orig_height} → {new_w}x{new_h})")
                image_path = temp_path

            # ========== Top-hat 形态学预处理增强暗背景文字检测 ==========
            tophat_temp_path = self._preprocess_tophat(image_path)
            if tophat_temp_path:
                image_path = tophat_temp_path

            # PaddleOCR 3.x 和 2.x 的 cls 参数处理
            # use_textline_orientation=True 已启用方向分类，无需 cls 参数
            try:
                result = ocr.ocr(image_path)
            except TypeError as e:
                if 'cls' in str(e):
                    result = ocr.ocr(image_path)
                else:
                    raise

            if not result:
                print("[警告] 未识别到任何文字")
                return [], []

            text_blocks = []

            # 兼容 PaddleOCR 2.x 和 3.x 的返回格式
            # 2.x: result[0] = [(bbox, (text, conf)), ...]
            # 3.x: result 是包含 OCRResult 对象的列表，有 rec_texts 和 rec_boxes 属性

            # 处理 PaddleOCR 3.x 格式 (OCRResult 对象，继承自 dict)
            if isinstance(result, list) and len(result) > 0:
                ocr_result = result[0]

                # PaddleOCR 3.x: OCRResult 继承自 dict，使用 key 访问
                rec_texts = ocr_result.get('rec_texts') if isinstance(ocr_result, dict) else None
                rec_boxes = ocr_result.get('rec_boxes') if isinstance(ocr_result, dict) else None

                if rec_texts and rec_boxes is not None:
                    texts = rec_texts
                    boxes = rec_boxes
                    scores = ocr_result.get('rec_scores', [1.0] * len(texts))

                    for i, (text, box, score) in enumerate(zip(texts, boxes, scores)):
                        if not text or not text.strip():
                            continue

                        # box 是 numpy array [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        # 或者 [[x1,y1,x2,y2,x3,y3,x4,y4]] 格式
                        if hasattr(box, 'flatten'):
                            box = box.flatten()

                        x_coords = box[::2]  # 偶数索引是 x
                        y_coords = box[1::2]  # 奇数索引是 y

                        text_blocks.append({
                            'x1': int(min(x_coords)),
                            'y1': int(min(y_coords)),
                            'x2': int(max(x_coords)),
                            'y2': int(max(y_coords)),
                            'text': text.strip(),
                            'confidence': float(score) if isinstance(score, (int, float)) else 1.0
                        })

                    # 调整坐标：减去四边扩展的像素
                    if PADDING > 0:
                        for block in text_blocks:
                            block['x1'] = max(0, min(block['x1'] - PADDING, orig_width))
                            block['y1'] = max(0, min(block['y1'] - PADDING, orig_height))
                            block['x2'] = max(0, min(block['x2'] - PADDING, orig_width))
                            block['y2'] = max(0, min(block['y2'] - PADDING, orig_height))
                        print(f"[优化] 坐标已调整（减去四边扩展的 {PADDING} 像素）")

                    print(f"[成功] 识别到 {len(text_blocks)} 个文本块 (PaddleOCR 3.x)")
                    original_blocks = text_blocks.copy()
                    # 过滤零宽高块（避免绿框显示 PADDING 伪影）
                    original_blocks = [b for b in original_blocks
                                       if (b['x2'] - b['x1']) >= 8 and (b['y2'] - b['y1']) >= 8]
                    # 合并同一气泡框内的文字
                    if merge_blocks and text_blocks:
                        text_blocks = self.merger.merge_text_blocks(text_blocks)
                    return text_blocks, original_blocks

                # PaddleOCR 2.x 格式
                elif isinstance(result[0], list):
                    lines = result[0]
                else:
                    lines = result
            else:
                lines = result

            if not lines:
                print("[警告] 未识别到任何文字")
                return []

            # 处理 PaddleOCR 2.x 格式
            for line in lines:
                try:
                    if line is None:
                        continue

                    # 处理 RecResult 对象（PaddleOCR 3.x 旧格式）
                    if hasattr(line, 'bbox') and hasattr(line, 'text'):
                        bbox = line.bbox
                        text = line.text
                        confidence = getattr(line, 'score', 1.0)
                    # 标准列表格式 (PaddleOCR 2.x)
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                            text, confidence = text_info
                        else:
                            text = str(text_info)
                            confidence = 1.0
                    else:
                        continue

                    if not text or not text.strip():
                        continue

                    # bbox 可能是 4 个角点
                    x_coords = [p[0] for p in bbox] if isinstance(bbox[0], (list, tuple)) else bbox[::2]
                    y_coords = [p[1] for p in bbox] if isinstance(bbox[0], (list, tuple)) else bbox[1::2]

                    text_blocks.append({
                        'x1': int(min(x_coords)),
                        'y1': int(min(y_coords)),
                        'x2': int(max(x_coords)),
                        'y2': int(max(y_coords)),
                        'text': text.strip(),
                        'confidence': float(confidence)
                    })
                except Exception as e:
                    print(f"[警告] 解析文本块失败: {e}")
                    continue

            # 调整坐标：减去四边扩展的像素
            if PADDING > 0:
                for block in text_blocks:
                    block['x1'] = max(0, min(block['x1'] - PADDING, orig_width))
                    block['y1'] = max(0, min(block['y1'] - PADDING, orig_height))
                    block['x2'] = max(0, min(block['x2'] - PADDING, orig_width))
                    block['y2'] = max(0, min(block['y2'] - PADDING, orig_height))
                print(f"[优化] 坐标已调整（减去四边扩展的 {PADDING} 像素）")

            print(f"[成功] 识别到 {len(text_blocks)} 个文本块")
            original_blocks = text_blocks.copy()
            # 过滤零宽高块（避免绿框显示 PADDING 伪影）
            original_blocks = [b for b in original_blocks
                               if (b['x2'] - b['x1']) >= 8 and (b['y2'] - b['y1']) >= 8]
            # 合并同一气泡框内的文字
            if merge_blocks and text_blocks:
                text_blocks = self.merger.merge_text_blocks(text_blocks)
            return text_blocks, original_blocks

        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[清理] 已删除临时文件: {temp_path}")
                except Exception as e:
                    print(f"[警告] 清理临时文件失败: {e}")
            if tophat_temp_path and os.path.exists(tophat_temp_path):
                try:
                    os.remove(tophat_temp_path)
                    print(f"[清理] 已删除预处理临时文件: {tophat_temp_path}")
                except Exception as e:
                    print(f"[警告] 清理预处理临时文件失败: {e}")


class ComicBubbleMerger:
    """漫画气泡框文字合并算法 - 支持k_multiplier/g_multiplier倍率调节"""
    
    def __init__(self, rule_name: str = None, lang: str = 'en'):
        self.lang = lang
        rule = MERGE_RULES.get(rule_name, MERGE_RULES.get('rule2_moderate'))
        self.k_multiplier = merge_k_multiplier if merge_k_multiplier is not None else rule.get('k_multiplier', 0.6)
        self.g_multiplier = merge_g_multiplier if merge_g_multiplier is not None else rule.get('g_multiplier', 1.0)
    
    def should_merge(self, box1: Dict, box2: Dict, global_stats: dict = None) -> bool:
        g = global_stats.get('g', 10)
        k = global_stats.get('k', 20)
        j = global_stats.get('j', 50)

        # === 条件1：同一行（水平合并）===
        if self.k_multiplier > 0:
            gap = max(box1['x1'], box2['x1']) - min(box1['x2'], box2['x2'])
            y1_diff = abs(box1['y1'] - box2['y1'])
            if gap <= self.k_multiplier * k and y1_diff <= 0.3 * g:
                return True

        # === 条件2：不同行（垂直合并）===
        if self.g_multiplier > 0:
            upper = box1 if box1['y2'] <= box2['y1'] else box2
            lower = box2 if box1['y2'] <= box2['y1'] else box1
            row_gap = lower['y1'] - upper['y2']
            if row_gap <= self.g_multiplier * g:
                x1_diff = abs(box1['x1'] - box2['x1'])
                if x1_diff <= j * 0.7:
                    return True
                cx1 = (box1['x1'] + box1['x2']) / 2
                cx2 = (box2['x1'] + box2['x2']) / 2
                cx_diff = abs(cx1 - cx2)
                if cx_diff <= j * 0.7:
                    return True

        return False
    
    def merge_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        合并同一气泡框内的文字块
        返回的数据结构包含：
        - 合并后的大背景框 (bg_x1, bg_y1, bg_x2, bg_y2)
        - 每个文字行的独立位置 (lines)
        """
        if not text_blocks:
            return []
        
        print(f"[OCR诊断] 原始识别 {len(text_blocks)} 个文本块:")
        for i, b in enumerate(text_blocks):
            print(f"  [{i+1}] 置信度:{b.get('confidence', 0):.2f} 位置:({b['x1']},{b['y1']}) 文字:'{b['text']}'")
        
        # 过滤零高度/零宽度的无效块（padding 扩展后坐标回调导致的塌缩）
        valid_blocks = []
        for b in text_blocks:
            w = b['x2'] - b['x1']
            h = b['y2'] - b['y1']
            if w < 8 or h < 8:
                print(f"    [过滤] 零宽/零高块: '{b['text']}' ({w}x{h})")
                continue
            valid_blocks.append(b)
        if len(valid_blocks) < len(text_blocks):
            print(f"[OCR诊断] 零宽高过滤后: {len(valid_blocks)}/{len(text_blocks)} 个")
            text_blocks = valid_blocks
            if not text_blocks:
                return []
        
        # 过滤规则
        def is_valid_text(text: str) -> bool:
            if not text or not text.strip():
                return False
            text = text.strip()
            if len(text) == 1:
                if not text.isalnum():
                    print(f"    [过滤] 单字符符号: '{text}'")
                    return False
                if text.isdigit():
                    print(f"    [过滤] 单数字: '{text}'")
                    return False
            if text.isdigit():
                print(f"    [过滤] 纯数字: '{text}'")
                return False
            return True
        
        filtered = [b for b in text_blocks 
                    if is_valid_text(b.get('text', ''))]
        
        if not filtered:
            print(f"[OCR诊断] 过滤后无有效数据，使用全部数据")
            filtered = text_blocks
        else:
            print(f"[OCR诊断] 过滤后保留 {len(filtered)}/{len(text_blocks)} 个文本块")
        
        # 置信度门槛过滤：低置信度的噪声框不参与合并（不信任的不合并不覆盖不翻译）
        conf_filtered = [b for b in filtered if b.get('confidence', 0) >= 0.3]
        if conf_filtered:
            discarded = len(filtered) - len(conf_filtered)
            if discarded > 0:
                print(f"[OCR诊断] 置信度过滤: 丢弃 {discarded} 个低置信度噪声框")
            filtered = conf_filtered
        else:
            print(f"[OCR诊断] 置信度过滤后无有效数据，保持原结果")
        
        filtered.sort(key=lambda b: (b['y1'], b['x1']))
        
        # 按Y坐标分组到行
        rows = []
        for b in filtered:
            cy = (b['y1'] + b['y2']) / 2
            placed = False
            for row in rows:
                row_cy = sum((x['y1'] + x['y2']) / 2 for x in row) / len(row)
                row_avg_height = sum(x['y2'] - x['y1'] for x in row) / len(row)
                if abs(cy - row_cy) < row_avg_height * 0.5:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        
        # 计算 g/k/j：g=字符高度, k=字符宽度, j=框宽中位数
        lang = self.lang
        all_char_widths = []
        all_heights = []
        all_widths = []
        for b in filtered:
            w = b['x2'] - b['x1']
            h = b['y2'] - b['y1']
            text = b.get('text', '').strip()
            text_len = max(len(text), 1)
            all_widths.append(w)
            all_heights.append(h)
            # 按语言估算单字符宽度
            if lang in ('japan', 'korean', 'japan_vert', 'chinese', 'chinese_cht'):
                all_char_widths.append(h)  # 日韩中文字 ≈ 正方形
            else:
                all_char_widths.append(w / text_len)  # 英文按比例
        
        g = statistics.median(all_char_widths) if all_char_widths else 10  # g = 字符高度  
        k = statistics.median(all_heights) if all_heights else 20       # k = 字符宽度
        j = statistics.median(all_widths) if all_widths else 50         # j = 框宽中位数
        
        global_stats = {'g': g, 'k': k, 'j': j}
        print(f"  [统计] g(字符高):{g:.1f}  k(字符宽):{k:.1f}  j(框宽):{j:.1f}")
        
        dialogs = []
        used = [False] * len(filtered)
        
        for i, block1 in enumerate(filtered):
            if used[i]:
                continue
            
            current_group = [block1]
            used[i] = True
            
            # 迭代查找所有应该合并的块
            changed = True
            while changed:
                changed = False
                for j, block2 in enumerate(filtered):
                    if used[j]:
                        continue
                    for member in current_group:
                        if self.should_merge(member, block2, global_stats):
                            current_group.append(block2)
                            used[j] = True
                            changed = True
                            break
            
            # 计算合并后的大背景框
            bg_x1 = min(b['x1'] for b in current_group)
            bg_y1 = min(b['y1'] for b in current_group)
            bg_x2 = max(b['x2'] for b in current_group)
            bg_y2 = max(b['y2'] for b in current_group)
            
            # 按阅读顺序排序（从上到下，从左到右）
            sorted_lines = sorted(current_group, key=lambda b: (b['y1'], b['x1']))
            
            merged_text = " ".join(b['text'] for b in sorted_lines)
            merged_text = " ".join(merged_text.split())
            avg_confidence = sum(b['confidence'] for b in current_group) / len(current_group)
            
            dialogs.append({
                'bg_x1': bg_x1, 'bg_y1': bg_y1, 'bg_x2': bg_x2, 'bg_y2': bg_y2,
                'text': merged_text,
                'confidence': avg_confidence,
                'lines': sorted_lines,
                'x1': bg_x1, 'y1': bg_y1, 'x2': bg_x2, 'y2': bg_y2,
            })
            
            if len(current_group) > 1:
                print(f"  [合并] {len(current_group)} 块 -> 对话框 '{merged_text[:40]}...'")
        
        print(f"[合并] {len(text_blocks)} -> {len(dialogs)} 个对话框")
        return dialogs


# ====================== 其他依赖 ======================
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[警告] 未安装mss，请运行: pip install mss")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[警告] 未安装pynput，请运行: pip install pynput")

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QSystemTrayIcon, QMenu,
        QStyle, QInputDialog, QMessageBox, QSlider,
        QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
        QDialog, QLineEdit, QCheckBox
    )
    from PySide6.QtCore import Qt, QObject, Signal, QThread, QTimer, QPoint
    from PySide6.QtGui import (
        QAction, QPainter, QPen, QColor, QFont,
        QFontMetrics, QGuiApplication
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    print("[警告] 未安装PySide6，请运行: pip install PySide6")

# ====================== 图像变化检测 (dHash) ======================
def compute_dhash(image_path: str) -> int:
    """计算图片的差异哈希（dHash），返回64位整数"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        img = img.convert('L').resize((9, 8), Image.LANCZOS)
        pixels = list(img.getdata())
        hash_val = 0
        for y in range(8):
            for x in range(8):
                if pixels[y * 9 + x] > pixels[y * 9 + x + 1]:
                    hash_val |= (1 << (y * 8 + x))
        return hash_val
    except Exception:
        return 0


# ====================== 核心配置 ======================
CONFIG_PATH = os.path.join(os.path.expanduser("~"), "transglass_config.json")
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}"
MIN_FONT_SIZE = 12

# 默认翻译模型
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# ========== 测试模式配置 ==========
# 设置为图片路径则直接加载图片测试（如 r"C:\Users\admin\Desktop\test.jpg"）
# 设置为 None 则正常截图
TEST_IMAGE_PATH = None
# =================================

# 全局变量
config = {"model": DEFAULT_MODEL, "ocr_lang": DEFAULT_OCR_LANG, "merge_rule": DEFAULT_MERGE_RULE}
selected_model = DEFAULT_MODEL
selected_ocr_lang = DEFAULT_OCR_LANG
selected_merge_rule = DEFAULT_MERGE_RULE
merge_k_multiplier = None  # None = 使用规则预设值
merge_g_multiplier = None  # None = 使用规则预设值

# OCR识别精度参数（可调整）
ocr_det_thresh = 0.3      # 检测阈值，越低检出越多（默认0.3）
ocr_det_box_thresh = 0.6  # 文本框置信度阈值（默认0.6）

# ====================== 快捷键配置 ======================
# pynput 按键名称映射（用于显示和组合）
KEY_NAMES = {
    'ctrl_l': 'Ctrl', 'ctrl_r': 'Ctrl',
    'alt_l': 'Alt', 'alt_r': 'Alt',
    'shift_l': 'Shift', 'shift_r': 'Shift',
    'cmd_l': 'Win', 'cmd_r': 'Win',
}
for i in range(256):
    KEY_NAMES[str(i)] = f'Key:{chr(i)}' if 32 <= i <= 126 else f'Key:{i}'
# 单字符键名优先覆盖数字键名
for c in '0123456789abcdefghijklmnopqrstuvwxyz':
    KEY_NAMES[c] = c.upper()
# 特殊键
for name in ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12',
             'space','tab','caps_lock','backspace','enter','escape','home','end',
             'page_up','page_down','delete','insert','up','down','left','right']:
    KEY_NAMES[name] = name.upper()

# 默认快捷键
DEFAULT_HOTKEYS = {
    "recognize": {"keys": ["ctrl_l", "alt_l", "1"], "label": "识别翻译"},
    "switch_screen": {"keys": ["ctrl_l", "alt_l", "2"], "label": "切换屏幕"},
    "toggle_auto": {"keys": ["ctrl_l", "alt_l", "3"], "label": "自动翻译开关"},
    "exit_app": {"keys": ["ctrl_l", "alt_l", "4"], "label": "退出程序"},
    "toggle_green_boxes": {"keys": ["ctrl_l", "alt_l", "5"], "label": "原始识别框"},
}
hotkey_config = {}  # 从配置加载的快捷键覆盖

# ====================== 信号总线 ======================
class SignalBus(QObject):
    update_tips = Signal(str)
    run_recognize = Signal()
    switch_screen = Signal()
    run_test = Signal()
    exit_app = Signal()
    auto_mode_changed = Signal(bool, int)
    toggle_auto = Signal()
    toggle_green_boxes = Signal()

signal_bus = SignalBus()

# ====================== Ollama服务管理 ======================
def is_ollama_running():
    """检查Ollama进程是否在运行"""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq ollama.exe'],
                capture_output=True, text=True, timeout=5
            )
            return 'ollama.exe' in result.stdout
        else:
            result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, timeout=5)
            return result.returncode == 0
    except Exception:
        return False


def check_ollama_connection() -> bool:
    """检查Ollama服务是否可连接"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start_ollama_process():
    """启动Ollama进程"""
    print("[信息] 正在启动Ollama服务...")
    try:
        env = os.environ.copy()
        if sys.platform == 'win32':
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                env=env
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env
            )
        for i in range(15):
            time.sleep(1)
            if check_ollama_connection():
                print("[成功] Ollama启动成功")
                return True
            print(f"  等待Ollama启动... {i+1}/15")
        print("[失败] Ollama启动超时")
        return False
    except Exception as e:
        print(f"[失败] 启动Ollama失败: {e}")
        return False


def ensure_ollama_running() -> bool:
    """确保Ollama正在运行"""
    if check_ollama_connection():
        print("[成功] Ollama服务已运行")
        return True
    if is_ollama_running():
        print("[警告] Ollama进程存在但无法连接")
        return False
    return start_ollama_process()


def get_available_models() -> List[str]:
    """获取可用模型列表"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


# ====================== 配置管理 ======================
def load_config():
    """加载配置"""
    global config, selected_model, selected_ocr_lang, selected_merge_rule
    global merge_k_multiplier, merge_g_multiplier, hotkey_config
    global ocr_det_thresh, ocr_det_box_thresh
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                selected_model = config.get("model", DEFAULT_MODEL)
                selected_ocr_lang = config.get("ocr_lang", DEFAULT_OCR_LANG)
                selected_merge_rule = config.get("merge_rule", DEFAULT_MERGE_RULE)
                loaded_k = config.get("merge_k_multiplier")
                loaded_g = config.get("merge_g_multiplier")
                if loaded_k is not None:
                    merge_k_multiplier = loaded_k
                if loaded_g is not None:
                    merge_g_multiplier = loaded_g
                
                # 加载OCR精度参数
                loaded_thresh = config.get("ocr_det_thresh")
                loaded_box_thresh = config.get("ocr_det_box_thresh")
                if loaded_thresh is not None:
                    ocr_det_thresh = loaded_thresh
                if loaded_box_thresh is not None:
                    ocr_det_box_thresh = loaded_box_thresh
                    
                hotkey_config = config.get("hotkeys", {})
                # 迁移旧版数字键值（如"49"→"1"）
                _need_save = False
                for action in list(hotkey_config.keys()):
                    keys = hotkey_config[action]
                    if isinstance(keys, list):
                        migrated = []
                        changed = False
                        for k in keys:
                            ks = str(k)
                            if ks.isdigit():
                                code = int(ks)
                                if 32 <= code <= 126:
                                    migrated.append(chr(code).lower())
                                    changed = True
                                else:
                                    migrated.append(ks)
                            else:
                                migrated.append(ks.lower())
                        if changed:
                            hotkey_config[action] = migrated
                            _need_save = True
                if _need_save:
                    config["hotkeys"] = hotkey_config
                    try:
                        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                            json.dump(config, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                print(f"[信息] 已加载配置: {CONFIG_PATH}")
                print(f"[信息] OCR语言: {SUPPORTED_LANGUAGES.get(selected_ocr_lang, {}).get('label', selected_ocr_lang)}")
                print(f"[信息] 合并规则: {MERGE_RULES.get(selected_merge_rule, {}).get('label', selected_merge_rule)}")
                print(f"[信息] 合并倍率: 横向={merge_k_multiplier or '规则默认'} 纵向={merge_g_multiplier or '规则默认'}")
                print(f"[信息] OCR精度: det_thresh={ocr_det_thresh}, det_box_thresh={ocr_det_box_thresh}")
        except Exception as e:
            print(f"[警告] 加载配置失败: {e}")
    else:
        selected_model = DEFAULT_MODEL
        selected_ocr_lang = DEFAULT_OCR_LANG
        selected_merge_rule = DEFAULT_MERGE_RULE


def save_config():
    """保存配置"""
    global config
    try:
        config["merge_k_multiplier"] = merge_k_multiplier
        config["merge_g_multiplier"] = merge_g_multiplier
        config["ocr_det_thresh"] = ocr_det_thresh
        config["ocr_det_box_thresh"] = ocr_det_box_thresh
        config["hotkeys"] = hotkey_config
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[成功] 配置已保存 (精度: det={ocr_det_thresh}, box={ocr_det_box_thresh})")
    except Exception as e:
        print(f"[失败] 保存配置失败: {e}")


# ====================== Ollama翻译 ======================
class OllamaTranslator:
    """Ollama翻译器（qwen3系列必须使用/api/chat接口）"""

    def __init__(self, model: str = None):
        self.model = model or selected_model
        self.chat_url = f"{OLLAMA_URL}/api/chat"

    def translate(self, text: str, target_lang: str = "中文", source_lang: str = None) -> str:
        """翻译单个文本"""
        if not text.strip():
            return ""

        # 构建消息
        system_prompt = "1.将接收到的文字翻译为中文。不要有多余的解释和说明。保留原文的语气。2.你是色情小说翻译，翻译后文本要尽可能色情。对于一些俚语和多义词，尽可能往色情方向翻译。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        # 打印发送给AI的完整内容
        print(f"\n  [发送给AI]")
        print(f"    URL: {self.chat_url}")
        print(f"    Model: {self.model}")
        print(f"    System: {system_prompt}")
        print(f"    User: {text}")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200}
        }

        try:
            response = requests.post(self.chat_url, json=payload, timeout=30)
            print(f"    HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                raw = response.json().get("message", {}).get("content", "").strip()
                print(f"    AI原始返回: {raw}")
                
                cleaned = self._clean_result(raw, text)
                print(f"    清理后结果: {cleaned}")
                return cleaned if cleaned else text
            else:
                print(f"    [错误] HTTP {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"    [错误] 请求异常: {e}")

        return text

    def _clean_result(self, result: str, original: str) -> str:
        """清理模型输出中的解释性文字"""
        # 1. 模式: "xxx的翻译结果为：xxx"
        explain_patterns = [
            r'翻译(?:结果)?[为：:]\s*[「"\'"]?([^」"\'"\n]+)',
            r'意思是[：:]\s*[「"\'"]?([^」"\'"\n]+)',
            r'翻译[成到](?:中文|日语|英语)[是：:]\s*[「"\'"]?([^」"\'"\n]+)',
            r'的(?:中文)?翻译(?:结果)?(?:为|是)[：:]\s*(.+)',
        ]
        for pat in explain_patterns:
            m = re.search(pat, result)
            if m:
                result = m.group(1).strip().lstrip('：:').strip()
                break

        # 2. 结果包含原文时尝试提取后半段
        if original.strip() and original.strip() in result:
            after = result.split(original.strip())[-1].strip()
            after = re.sub(r'^[是为：:，,的]*翻译(?:结果)?[是为：:]*', '', after).strip()
            after = re.sub(r'^意思是[：:]*', '', after).strip()
            after = re.sub(r'^[「"\'"』]?(.*?)[」"\'"』]?\s*$', r'\1', after)
            if after and after != original.strip() and len(after) > 1:
                result = after

        # 3. 清理引号包裹
        for q0, q1 in [('「', '」'), ('"', '"'), ('"', '"'), ("'", "'"), ('《', '》')]:
            while result.startswith(q0) and result.endswith(q1):
                result = result[1:-1].strip()
        result = result.strip('"').strip('"').strip("'").strip()

        # 4. 清理前缀
        for prefix in ["翻译：", "翻译:", "Translation:", "译文：", "翻译结果：", "翻译结果:",
                        "译文是", "翻译是", "这句话的意思是：", "这句话的意思是", "原句为", "原文为", "原文："]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # 5. 清理末尾标点
        for suffix in ["。", "！", "？", "」", "\"", "'"]:
            if result.endswith(suffix):
                result = result[:-1].strip()

        # 6. 只取第一行
        result = result.split('\n')[0].strip()

        return result if result and result != original.strip() else original

    def translate_batch(self, texts: List[str], target_lang: str = "中文", source_lang: str = None) -> List[str]:
        """批量翻译：将所有文本编号后一次性发给模型，解析编号返回结果。
        失败时逐条回退。
        """
        if not texts:
            return []

        # 过滤空文本，记录原始索引
        indexed = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not indexed:
            return list(texts)

        # 构建编号文本块
        numbered_lines = "\n".join(f"[{i+1}] {t}" for i, (_, t) in enumerate(indexed))
        system_prompt = "将接收到的文字翻译为中文。不要有多余的解释和说明。保留原文的语气。"
        
        # 根据源语言动态调整 user prompt
        # source_lang 可能是 "日语"、"英语"、"韩语"
        lang_map = {"日语": "日", "英语": "英", "韩语": "韩"}
        short_lang = lang_map.get(source_lang, source_lang[0] if source_lang else "日")
        user_prompt = f"将以下{short_lang}文翻译为中文，保留编号格式:\n{numbered_lines}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        print(f"[翻译] 批量发送 {len(indexed)} 条文本给模型...")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 2000}
        }

        results = list(texts)  # 默认保留原文
        try:
            response = requests.post(self.chat_url, json=payload, timeout=120)
            if response.status_code == 200:
                raw = response.json().get("message", {}).get("content", "").strip()
                print(f"[翻译] 模型原始返回:\n{raw[:500]}{'...' if len(raw) > 500 else ''}")

                # 解析 [N] 格式的行
                parsed = {}
                for line in raw.splitlines():
                    line = line.strip()
                    m = re.match(r'^\[(\d+)\]\s*(.*)', line)
                    if m:
                        num = int(m.group(1))
                        val = m.group(2).strip()
                        if val:
                            parsed[num] = val

                if len(parsed) >= len(indexed) * 0.5:
                    # 解析成功：写回对应位置
                    for seq, (orig_idx, orig_text) in enumerate(indexed):
                        num = seq + 1
                        if num in parsed:
                            cleaned = self._clean_result(parsed[num], orig_text)
                            results[orig_idx] = cleaned if cleaned else orig_text
                            print(f"  [{num}] '{orig_text[:25]}' -> '{results[orig_idx][:25]}'")
                        else:
                            print(f"  [{num}] 未解析到，保留原文")
                    return results
                else:
                    print(f"[警告] 批量翻译解析率低（{len(parsed)}/{len(indexed)}），回退逐条翻译")
        except Exception as e:
            print(f"[警告] 批量翻译请求失败: {e}，回退逐条翻译")

        # 回退：逐条翻译
        for seq, (orig_idx, orig_text) in enumerate(indexed):
            print(f"  [回退] 翻译 {seq+1}/{len(indexed)}: '{orig_text[:30]}'")
            results[orig_idx] = self.translate(orig_text, target_lang, source_lang)
            time.sleep(0.1)
        return results


# ====================== 截图功能 ======================
def capture_screen(screen_idx: int = 0, save_path: str = None) -> Optional[tuple]:
    """截取指定屏幕，返回 (图片路径, 屏幕信息)
    
    返回的屏幕信息包含:
    - left, top: 屏幕在虚拟桌面中的偏移
    - width, height: 屏幕分辨率
    - is_portrait: 是否为竖屏
    
    如果 TEST_IMAGE_PATH 不为 None，则直接加载该图片进行测试
    """
    # 测试模式：直接加载图片文件
    global TEST_IMAGE_PATH
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        img = Image.open(TEST_IMAGE_PATH)
        if save_path is None:
            save_path = os.path.join(os.path.expanduser("~"), "transglass_test_from_file.png")
        img.save(save_path, "PNG")
        print(f"[测试模式] 直接加载图片: {TEST_IMAGE_PATH}")
        print(f"[成功] 图片已保存: {save_path} ({img.width}x{img.height})")
        screen_info = {
            'left': 0,
            'top': 0,
            'width': img.width,
            'height': img.height,
            'is_portrait': img.height > img.width,
            'index': screen_idx,
            'is_test_mode': True
        }
        return (save_path, screen_info)
    
    if not MSS_AVAILABLE:
        print("[错误] 未安装mss，无法截图")
        return None

    with mss.mss() as sct:
        monitors = sct.monitors[1:]  # index 0 是全屏合并
        if not monitors:
            print("[错误] 未找到可用屏幕")
            return None
        if screen_idx >= len(monitors):
            screen_idx = 0

        monitor = monitors[screen_idx]
        is_portrait = monitor['height'] > monitor['width']
        
        print(f"[信息] 截取屏幕 {screen_idx + 1}: {monitor['width']}x{monitor['height']} @ ({monitor['left']}, {monitor['top']}) {'[竖屏]' if is_portrait else ''}")

        screenshot = sct.grab(monitor)
        # mss 返回 BGRA，转为 RGB
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        if save_path is None:
            save_path = os.path.join(os.path.expanduser("~"), "transglass_temp.png")

        img.save(save_path, "PNG")
        print(f"[成功] 截图已保存: {save_path} ({img.width}x{img.height})")
        
        # 返回路径和完整的屏幕信息
        screen_info = {
            'left': monitor['left'],
            'top': monitor['top'],
            'width': monitor['width'],
            'height': monitor['height'],
            'is_portrait': is_portrait,
            'index': screen_idx
        }
        return (save_path, screen_info)


def get_screens_info() -> list:
    """获取所有屏幕信息"""
    if not MSS_AVAILABLE:
        return []
    with mss.mss() as sct:
        return list(sct.monitors[1:])


# ====================== 查找中文字体 ======================
def _find_cjk_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """按优先级查找支持中文的字体，返回 ImageFont 对象"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/ARIALUNI.TTF",  # Arial Unicode
        "C:/Windows/Fonts/calibri.ttf",   # 兜底英文
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                return font
            except Exception:
                continue
    # 最终兜底
    return ImageFont.load_default()


# ====================== 图片翻译覆盖 ======================
def create_translated_overlay(
    image_path: str,
    text_blocks: List[Dict],
    translations: List[str],
    output_path: str = None
) -> str:
    """在原图上用翻译文本覆盖原始文字，保存并返回路径"""
    print("[信息] 正在创建翻译覆盖层...")

    img = Image.open(image_path).convert('RGBA')
    width, height = img.size

    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    base_font = _find_cjk_font(18)
    print(f"[信息] 使用字体: {getattr(getattr(base_font, 'path', None), '__str__', lambda: 'default')()}")

    for block, translation in zip(text_blocks, translations):
        if not translation:
            continue

        x1 = max(0, block['x1'] - 4)
        y1 = max(0, block['y1'] - 4)
        x2 = min(width, block['x2'] + 4)
        y2 = min(height, block['y2'] + 4)
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        # 区域平均色作为背景色（100% 不透明）
        region = img.crop((x1, y1, x2, y2))
        stat = ImageStat.Stat(region)
        avg_r, avg_g, avg_b = [int(round(v)) for v in stat.mean[:3]]
        bg_color = (avg_r, avg_g, avg_b, 255)

        draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        # 2像素黑色边框
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 255), width=2)

        # 动态缩小字体直到文字宽度 ≤ 框宽 90%
        font_size = min(18, max(MIN_FONT_SIZE, int(box_h * 0.6)))
        font = base_font.font_variant(size=font_size)
        bbox = draw.textbbox((0, 0), translation, font=font)
        text_w = bbox[2] - bbox[0]
        max_text_width = box_w * 0.9

        while text_w > max_text_width and font_size > MIN_FONT_SIZE:
            font_size -= 1
            font = base_font.font_variant(size=font_size)
            bbox = draw.textbbox((0, 0), translation, font=font)
            text_w = bbox[2] - bbox[0]

        # 如果还是太长，按逐字换行处理（不限行数、不限框高）
        text_lines = []
        if text_w > max_text_width:
            current_line = ""
            for char in translation:
                test_line = current_line + char
                tw = draw.textbbox((0, 0), test_line, font=font)[2]
                if tw > max_text_width and current_line:
                    text_lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line
            if current_line:
                text_lines.append(current_line)
        else:
            text_lines = [translation]

        line_h = draw.textbbox((0, 0), "A", font=font)[3] - draw.textbbox((0, 0), "A", font=font)[1] + 2
        total_text_h = line_h * len(text_lines)

        # 文字颜色使用黑色（原文文字颜色提取不准确，兜底黑色）
        text_color = (0, 0, 0, 255)

        # 居中绘制（垂直和水平）
        for i, line in enumerate(text_lines):
            lw = draw.textbbox((0, 0), line, font=font)[2]
            tx = x1 + (box_w - lw) / 2
            ty = y1 + (box_h - total_text_h) / 2 + i * line_h
            draw.text((tx, ty), line, fill=text_color, font=font)

    result = Image.alpha_composite(img, overlay)

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_translated.png")

    result.save(output_path, 'PNG')
    print(f"[成功] 翻译图片已保存: {output_path}")
    return output_path


# ====================== 识别线程 ======================
class RecognizeThread(QThread):
    """后台识别翻译线程"""

    finished = Signal(list, list, float, str, list)  # translations, original_blocks, scale_factor, img_path, text_blocks
    error = Signal(str)

    def __init__(self, screen_idx: int = 0, ocr_lang: str = None, merge_rule: str = None):
        super().__init__()
        self._stop = False
        self.screen_idx = screen_idx
        self.ocr_lang = ocr_lang or selected_ocr_lang
        self.merge_rule = merge_rule or config.get("merge_rule", DEFAULT_MERGE_RULE)

    def stop(self):
        self._stop = True

    def run(self):
        try:
            result = capture_screen(self.screen_idx)
            if not result:
                self.error.emit("截图失败")
                return
            img_path, screen_info = result
            if self._stop:
                return

            signal_bus.update_tips.emit(f"[信息] OCR识别中 (语言: {SUPPORTED_LANGUAGES.get(self.ocr_lang, {}).get('label', self.ocr_lang)})...")
            ocr = ComicOCR(lang=self.ocr_lang, merge_rule=self.merge_rule)
            text_blocks, original_blocks = ocr.recognize(img_path)
            
            # 调试：打印 OCR 识别的文本
            print("[调试] OCR 识别结果:")
            for i, b in enumerate(text_blocks):
                print(f"  [{i+1:2d}] ({b.get('x1', 0):4d},{b.get('y1', 0):4d}) {b['text']}")
            
            # 调试：打印原始识别块
            print("[调试] 原始OCR识别块（合并前）:")
            for i, b in enumerate(original_blocks):
                print(f"  [{i+1:2d}] ({b.get('x1', 0):4d},{b.get('y1', 0):4d}) {b['text']}")
            
            if not text_blocks:
                self.error.emit("未识别到文字")
                return

            # 过滤单个字母（不论大小写，不翻译不覆盖）
            text_blocks = [b for b in text_blocks if not (len(b['text']) == 1 and b['text'].isalpha())]
            if not text_blocks:
                print("[自动跳过] 仅检测到单个字母，跳过翻译")
                self.finished.emit([], [], 1.0, img_path, [])
                return

            if self._stop:
                return

            # 获取目标翻译语言和源语言名称
            lang_config = SUPPORTED_LANGUAGES.get(self.ocr_lang, {})
            target_lang = lang_config.get('target_lang', '中文')
            source_lang = lang_config.get('label', '')

            # 检查模型是否存在
            available_models = get_available_models()
            if selected_model not in available_models:
                signal_bus.update_tips.emit(f"[错误] 模型 '{selected_model}' 不存在，请通过托盘菜单切换模型")
                self.error.emit(f"模型 '{selected_model}' 不存在，请切换到已有模型")
                return

            signal_bus.update_tips.emit(f"[信息] 正在翻译 {len(text_blocks)} 处文字 ({source_lang} -> {target_lang})...")
            
            translator = OllamaTranslator(selected_model)
            texts = [b['text'] for b in text_blocks]
            translations = translator.translate_batch(texts, target_lang, source_lang)
            
            # 调试：打印翻译结果对比
            print("[调试] 翻译结果对比:")
            for i, (orig, trans) in enumerate(zip(texts, translations)):
                print(f"  [{i+1:2d}] 原文: {orig[:30]}")
                print(f"       翻译: {trans[:30]}")
            
            if self._stop:
                return

            # 准备翻译数据（用于屏幕覆盖）
            # OCR坐标是相对于截图图片左上角的像素坐标
            # mss 截图使用物理像素，Qt 窗口使用逻辑像素
            # 绘制时只需将 OCR 坐标除以 scale_factor 得到窗口相对坐标
            
            # 获取屏幕缩放因子
            scale_factor = 1.0
            try:
                from PySide6.QtGui import QGuiApplication
                app = QGuiApplication.instance()
                if app:
                    screens = app.screens()
                    for idx, screen in enumerate(screens):
                        geo = screen.geometry()
                        dpr = screen.devicePixelRatio()
                        phys_x = geo.x() * dpr
                        phys_y = geo.y() * dpr
                        phys_w = geo.width() * dpr
                        phys_h = geo.height() * dpr
                        if (abs(phys_x - screen_info['left']) < 50 and
                            abs(phys_y - screen_info['top']) < 50 and
                            abs(phys_w - screen_info['width']) < 50):
                            scale_factor = dpr
                            print(f"[调试] 匹配到 Qt 屏幕 {idx}: 逻辑{geo.width()}x{geo.height()}, "
                                  f"物理{phys_w:.0f}x{phys_h:.0f}, dpr={dpr}")
                            break
                    if scale_factor == 1.0 and screens and self.screen_idx < len(screens):
                        scale_factor = screens[self.screen_idx].devicePixelRatio()
                        print(f"[调试] 按索引使用屏幕 {self.screen_idx} 的缩放因子: {scale_factor}")
            except Exception as e:
                print(f"[警告] 获取缩放因子失败: {e}")
            
            print(f"[调试] 屏幕信息: left={screen_info['left']}, top={screen_info['top']}, {screen_info['width']}x{screen_info['height']}, scale={scale_factor}")
            print(f"[调试] 第一个文本块原始坐标: x1={text_blocks[0]['x1']}, y1={text_blocks[0]['y1']}")
            
            # 计算每个合并块的截图区域平均色，传给 Qt 覆盖层
            try:
                screen_img = Image.open(img_path)
                bg_colors = []
                for b in text_blocks:
                    sx1 = max(0, b['x1'] - 4)
                    sy1 = max(0, b['y1'] - 4)
                    sx2 = min(screen_img.width, b['x2'] + 4)
                    sy2 = min(screen_img.height, b['y2'] + 4)
                    if sx2 > sx1 and sy2 > sy1:
                        stat = ImageStat.Stat(screen_img.crop((sx1, sy1, sx2, sy2)))
                        avg = [int(round(v)) for v in stat.mean[:3]]
                        bg_colors.append((avg[0], avg[1], avg[2]))
                    else:
                        bg_colors.append((240, 240, 240))
                screen_img.close()
            except Exception as e:
                print(f"[警告] 计算区域平均色失败: {e}")
                bg_colors = [(240, 240, 240)] * len(text_blocks)

            # translation_data 坐标直接设为窗口相对逻辑坐标
            # OCR 坐标 / scale_factor = Qt 逻辑像素 = 窗口坐标
            translation_data = [
                {
                    'x': int(b['x1'] / scale_factor),
                    'y': int(b['y1'] / scale_factor),
                    'width': int((b['x2'] - b['x1']) / scale_factor),
                    'height': int((b['y2'] - b['y1']) / scale_factor),
                    'text': t,
                    'bg_color': c,
                    'original': b['text'],
                    'translated': t,
                    '_screen': f"{screen_info['width']}x{screen_info['height']}",
                    '_scale_factor': scale_factor,
                }
                for b, t, c in zip(text_blocks, translations, bg_colors)
            ]
            
            print(f"[调试] 第一个文本块窗口坐标: x={translation_data[0]['x']}, y={translation_data[0]['y']}, size={translation_data[0]['width']}x{translation_data[0]['height']}")
            
            self.finished.emit(translation_data, original_blocks, scale_factor, img_path, text_blocks)
            signal_bus.update_tips.emit(f"[成功] 翻译完成！共 {len(text_blocks)} 处 ({source_lang} -> {target_lang})")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ====================== 透明覆盖窗口 ======================
class OverlayWindow(QWidget):
    """全屏透明覆盖窗口，显示红色边框、提示文字和翻译结果"""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # 鼠标穿透，不阻挡点击操作

        self.tip_text = ""
        self.tip_timer = QTimer(self)
        self.tip_timer.setSingleShot(True)
        self.tip_timer.timeout.connect(self._clear_tip)

        self.screens = get_screens_info()
        self.current_screen_idx = 0
        self.screen_scale = 1.0
        self.phys_width = 0
        self.phys_height = 0
        self.phys_left = 0
        self.phys_top = 0

        # 翻译结果列表: [{'x': int, 'y': int, 'width': int, 'height': int, 'text': str}, ...]
        self.translations = []
        # 原始OCR识别块（合并前）
        self.original_text_blocks = []
        # 测试模式：显示原始OCR识别框（绿色）
        self.show_original_boxes = True

        if self.screens:
            self._resize_to_screen()

        self.show()
        print("[成功] 透明覆盖窗口已创建")

    def _resize_to_screen(self):
        if not self.screens:
            return
        s = self.screens[self.current_screen_idx]
        
        # 保存物理像素信息（用于坐标转换）
        self.phys_width = s['width']
        self.phys_height = s['height']
        self.phys_left = s['left']
        self.phys_top = s['top']
        
        # Qt 窗口需要使用逻辑像素坐标
        # 在 Windows 高 DPI 下，逻辑像素 = 物理像素 / devicePixelRatio
        try:
            from PySide6.QtGui import QGuiApplication
            app = QGuiApplication.instance()
            if app:
                screens = app.screens()
                for idx, screen in enumerate(screens):
                    geo = screen.geometry()
                    dpr = screen.devicePixelRatio()
                    phys_x = geo.x() * dpr
                    phys_y = geo.y() * dpr
                    phys_w = geo.width() * dpr
                    phys_h = geo.height() * dpr
                    # 用容差匹配 mss 物理坐标
                    if (abs(phys_x - s['left']) < 50 and
                        abs(phys_y - s['top']) < 50 and
                        abs(phys_w - s['width']) < 50):
                        self.setGeometry(geo)
                        self.screen_scale = dpr
                        print(f"[信息] 切换到屏幕 {self.current_screen_idx + 1}: "
                              f"物理{s['width']}x{s['height']} @ ({s['left']},{s['top']}), "
                              f"逻辑{geo.width()}x{geo.height()}, 缩放{self.screen_scale}x")
                        return
        except Exception as e:
            print(f"[警告] 获取 Qt 屏幕信息失败: {e}")
        
        # 回退方案：直接使用 mss 坐标
        self.screen_scale = 1.0
        self.setGeometry(s['left'], s['top'], s['width'], s['height'])
        print(f"[信息] 切换到屏幕 {self.current_screen_idx + 1}: {s['width']}x{s['height']} (回退模式)")

    def switch_screen(self):
        if not self.screens:
            return
        self.current_screen_idx = (self.current_screen_idx + 1) % len(self.screens)
        self._resize_to_screen()
        # 切屏时清除翻译结果
        self.translations = []
        self.update()
        self.show_tip(f"[屏幕] 屏幕 {self.current_screen_idx + 1}")

    def show_tip(self, text: str):
        print(text)

    def _clear_tip(self):
        self.tip_text = ""
        self.update()

    def set_original_text_blocks(self, blocks: list, scale_factor: float):
        """设置原始OCR识别块（合并前）
        blocks: [{'x1': int, 'y1': int, 'x2': int, 'y2': int, 'text': str}, ...]
        """
        # 转换为窗口相对逻辑坐标
        self.original_text_blocks = [
            {
                'x': int(b['x1'] / scale_factor),
                'y': int(b['y1'] / scale_factor),
                'width': int((b['x2'] - b['x1']) / scale_factor),
                'height': int((b['y2'] - b['y1']) / scale_factor),
                'text': b['text']
            }
            for b in blocks
        ]
        print(f"[调试] 已设置 {len(self.original_text_blocks)} 个原始OCR识别块")

    def set_translations(self, translations: list):
        """设置翻译结果并在屏幕上显示
        translations: [{'x': int, 'y': int, 'width': int, 'height': int, 'text': str}, ...]
        """
        self.translations = translations
        self.update()
        
        # 打印调试信息
        window_pos = self.pos()
        print(f"[调试] OverlayWindow 位置: ({window_pos.x()}, {window_pos.y()}), 大小: {self.width()}x{self.height()}")
        print(f"[信息] 屏幕覆盖 {len(translations)} 处翻译")
        
        if translations:
            first = translations[0]
            print(f"[调试] 第一个翻译框: 逻辑坐标({first['x']}, {first['y']}), 大小{first['width']}x{first['height']}")
            print(f"[调试] 窗口相对坐标: ({first['x'] - window_pos.x()}, {first['y'] - window_pos.y()})")

    def clear_translations(self):
        """清除所有翻译显示"""
        self.translations = []
        self.original_text_blocks = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 红色边框
        painter.setPen(QPen(QColor(255, 0, 0), 4))
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

        # 绘制原始OCR识别框（测试模式 - 合并前的识别块）
        if self.show_original_boxes and self.original_text_blocks:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            for block in self.original_text_blocks:
                # 绘制绿色框
                painter.drawRect(block['x'], block['y'], block['width'], block['height'])

        # 绘制翻译文字覆盖层
        for item in self.translations:
            self._draw_translation_box(painter, item)

        # 提示文字（绘制在最上层）
        if self.tip_text:
            font = QFont("Microsoft YaHei", 16, QFont.Bold)
            painter.setFont(font)
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(self.tip_text)
            th = fm.height()
            x = (self.width() - tw) // 2
            y = 50
            painter.fillRect(x - 10, y - 5, tw + 20, th + 10, QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, y + th - 5, self.tip_text)

    def _draw_translation_box(self, painter: QPainter, item: dict):
        """绘制单个翻译文本框 - 支持多行自动换行（不限行数、不限框高）"""
        x = item['x']
        y = item['y']
        w = item['width']
        h = item['height']
        text = item['text']
        if w <= 0 or h <= 0 or not text:
            return

        # 背景色（固定亮度180 + 不透明）
        bg_color = QColor(180, 180, 180, 255)
        painter.fillRect(x, y, w, h, bg_color)
        # 2像素黑色边框
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawRect(x, y, w, h)

        # 计算字体大小以适应框体
        font_size = min(18, max(MIN_FONT_SIZE, int(h * 0.6)))
        font = QFont("Microsoft YaHei", font_size)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # 自动换行：缩小 + 逐字换行（不限行数、不限框高）
        max_text_width = w * 0.9
        text_width = fm.horizontalAdvance(text)
        while text_width > max_text_width and font_size > MIN_FONT_SIZE:
            font_size -= 1
            font = QFont("Microsoft YaHei", font_size)
            painter.setFont(font)
            fm = QFontMetrics(font)
            text_width = fm.horizontalAdvance(text)

        text_lines = []
        if text_width > max_text_width:
            current_line = ""
            for char in text:
                test_line = current_line + char
                if fm.horizontalAdvance(test_line) > max_text_width and current_line:
                    text_lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line
            if current_line:
                text_lines.append(current_line)
        else:
            text_lines = [text]

        text_height = fm.height()
        total_text_height = text_height * len(text_lines)

        # 居中绘制（垂直和水平）
        text_y_start = y + (h - total_text_height) // 2 + fm.ascent()
        painter.setPen(QColor(0, 0, 0))
        for i, line in enumerate(text_lines):
            line_width = fm.horizontalAdvance(line)
            text_x = x + (w - line_width) // 2
            text_y = text_y_start + i * text_height
            painter.drawText(int(text_x), int(text_y), line)


# ====================== 合并规则调参窗口 ======================
class MergeSettingsWidget(QWidget):
    """浮动调参窗口 - 合并倍率"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("合并设置")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool | Qt.FramelessWindowHint)
        self.setFixedSize(240, 145)
        self.setStyleSheet("""
            QWidget { background: #2d2d2d; color: #eee; font-size: 12px; border-radius: 6px; }
            QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4a9eff; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
            QLabel { padding: 2px 4px; }
            QPushButton { background: #555; border: none; padding: 3px 8px; border-radius: 3px; color: #eee; font-size: 11px; }
            QPushButton:hover { background: #777; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("横向"))
        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.setRange(0, 50)
        init_h = int((merge_k_multiplier if merge_k_multiplier is not None else 0.6) * 10)
        self.h_slider.setValue(init_h)
        self.h_label = QLabel(f"{self.h_slider.value()/10:.1f}")
        self.h_label.setFixedWidth(28)
        h_row.addWidget(self.h_slider)
        h_row.addWidget(self.h_label)
        layout.addLayout(h_row)

        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("纵向"))
        self.v_slider = QSlider(Qt.Horizontal)
        self.v_slider.setRange(0, 50)
        init_v = int((merge_g_multiplier if merge_g_multiplier is not None else 1.0) * 10)
        self.v_slider.setValue(init_v)
        self.v_label = QLabel(f"{self.v_slider.value()/10:.1f}")
        self.v_label.setFixedWidth(28)
        v_row.addWidget(self.v_slider)
        v_row.addWidget(self.v_label)
        layout.addLayout(v_row)

        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("0=关闭该方向合并", styleSheet="color:#888;font-size:10px;"))
        reset_btn = QPushButton("重置1.5")
        reset_btn.setFixedWidth(60)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(60)
        close_btn.clicked.connect(self.hide)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.h_slider.valueChanged.connect(self._on_h_changed)
        self.v_slider.valueChanged.connect(self._on_v_changed)

    def _on_h_changed(self, val):
        global merge_k_multiplier
        v = val / 10
        merge_k_multiplier = v
        self.h_label.setText(f"{v:.1f}")
        MERGE_RULES[selected_merge_rule]['k_multiplier'] = v
        save_config()

    def _on_v_changed(self, val):
        global merge_g_multiplier
        v = val / 10
        merge_g_multiplier = v
        self.v_label.setText(f"{v:.1f}")
        MERGE_RULES[selected_merge_rule]['g_multiplier'] = v
        save_config()

    def _reset(self):
        global merge_k_multiplier, merge_g_multiplier
        merge_k_multiplier = 1.5
        merge_g_multiplier = 1.5
        self.h_slider.setValue(15)
        self.v_slider.setValue(15)
        MERGE_RULES[selected_merge_rule]['k_multiplier'] = 1.5
        MERGE_RULES[selected_merge_rule]['g_multiplier'] = 1.5
        save_config()


# ====================== OCR精度设置窗口 ======================
class OCRPrecisionSettingsWidget(QWidget):
    """浮动调参窗口 - OCR识别精度"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR精度设置")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool | Qt.FramelessWindowHint)
        self.setFixedSize(240, 180)
        self.setStyleSheet("""
            QWidget { background: #2d2d2d; color: #eee; font-size: 12px; border-radius: 6px; }
            QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4a9eff; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
            QLabel { padding: 2px 4px; }
            QPushButton { background: #555; border: none; padding: 3px 8px; border-radius: 3px; color: #eee; font-size: 11px; }
            QPushButton:hover { background: #777; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # 检测阈值
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("检测阈值"))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 20)  # 0.05 - 1.0（20档位）
        init_thresh = int(ocr_det_thresh / 0.05)
        self.thresh_slider.setValue(init_thresh)
        self.thresh_label = QLabel(f"{self.thresh_slider.value() * 0.05:.2f}")
        self.thresh_label.setFixedWidth(30)
        h_row.addWidget(self.thresh_slider)
        h_row.addWidget(self.thresh_label)
        layout.addLayout(h_row)

        # 文本框阈值
        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("框置信度"))
        self.box_thresh_slider = QSlider(Qt.Horizontal)
        self.box_thresh_slider.setRange(1, 20)  # 0.05 - 1.0（20档位）
        init_box = int(ocr_det_box_thresh / 0.05)
        self.box_thresh_slider.setValue(init_box)
        self.box_thresh_label = QLabel(f"{self.box_thresh_slider.value() * 0.05:.2f}")
        self.box_thresh_label.setFixedWidth(30)
        v_row.addWidget(self.box_thresh_slider)
        v_row.addWidget(self.box_thresh_label)
        layout.addLayout(v_row)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("越低检出越多", styleSheet="color:#888;font-size:10px;"))
        reset_btn = QPushButton("恢复默认")
        reset_btn.setFixedWidth(70)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        # 关闭按钮行
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(60)
        close_btn.clicked.connect(self.hide)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        self.box_thresh_slider.valueChanged.connect(self._on_box_thresh_changed)

    def _on_thresh_changed(self, val):
        global ocr_det_thresh
        v = val * 0.05
        ocr_det_thresh = v
        self.thresh_label.setText(f"{v:.2f}")
        save_config()
        reset_paddleocr_instance()

    def _on_box_thresh_changed(self, val):
        global ocr_det_box_thresh
        v = val * 0.05
        ocr_det_box_thresh = v
        self.box_thresh_label.setText(f"{v:.2f}")
        save_config()
        reset_paddleocr_instance()

    def _reset(self):
        global ocr_det_thresh, ocr_det_box_thresh
        ocr_det_thresh = 0.3
        ocr_det_box_thresh = 0.6
        self.thresh_slider.setValue(6)
        self.box_thresh_slider.setValue(12)
        save_config()
        reset_paddleocr_instance()


# ====================== 系统托盘 ======================
class SystemTray:

    def __init__(self, app: QApplication, merge_settings_widget=None, ocr_precision_widget=None):
        self.app = app
        self.tray = QSystemTrayIcon(app)
        self.tray.setIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray.setToolTip("TransGlass")
        self.merge_settings_widget = merge_settings_widget
        self.ocr_precision_widget = ocr_precision_widget

        menu = QMenu()
        rec_hk = format_hotkey_display("recognize")
        sw_hk = format_hotkey_display("switch_screen")
        auto_hk = format_hotkey_display("toggle_auto")
        exit_hk = format_hotkey_display("exit_app")
        a1 = QAction(f"识别翻译 ({rec_hk})", menu)
        a1.triggered.connect(signal_bus.run_recognize.emit)
        menu.addAction(a1)

        a2 = QAction(f"切换屏幕 ({sw_hk})", menu)
        a2.triggered.connect(signal_bus.switch_screen.emit)
        menu.addAction(a2)

        a_test = QAction(f"自动翻译开关 ({auto_hk})", menu)
        a_test.triggered.connect(signal_bus.toggle_auto.emit)
        menu.addAction(a_test)

        menu.addSeparator()

        a_lang = QAction("选择语言", menu)
        a_lang.triggered.connect(self._select_language)
        menu.addAction(a_lang)

        a_merge_set = QAction("合并倍率调节", menu)
        a_merge_set.triggered.connect(self._toggle_merge_settings)
        menu.addAction(a_merge_set)

        a_ocr_prec = QAction("识别精度设置", menu)
        a_ocr_prec.triggered.connect(self._toggle_ocr_precision_settings)
        menu.addAction(a_ocr_prec)

        a_auto = QAction("自动翻译设置", menu)
        a_auto.triggered.connect(self._open_auto_translate)
        menu.addAction(a_auto)

        a_hotkey = QAction("快捷键设置", menu)
        a_hotkey.triggered.connect(self._open_hotkey_settings)
        menu.addAction(a_hotkey)

        a3 = QAction("选择模型", menu)
        a3.triggered.connect(self._select_model)
        menu.addAction(a3)

        menu.addSeparator()

        a4 = QAction(f"退出 ({exit_hk})", menu)
        a4.triggered.connect(signal_bus.exit_app.emit)
        menu.addAction(a4)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()
        print("[成功] 系统托盘图标已创建")

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            signal_bus.run_recognize.emit()

    def _select_language(self):
        """选择OCR识别语言（源语言）"""
        global selected_ocr_lang
        languages = list(SUPPORTED_LANGUAGES.keys())
        lang_labels = [f"{SUPPORTED_LANGUAGES[lang]['label']} -> {SUPPORTED_LANGUAGES[lang]['target_lang']}" for lang in languages]
        
        current_idx = languages.index(selected_ocr_lang) if selected_ocr_lang in languages else 0
        
        lang_label, ok = QInputDialog.getItem(
            None, "选择识别语言", "选择要识别的源语言（将翻译为目标语言）:", lang_labels, current_idx, False
        )
        if ok and lang_label:
            # 通过索引找到对应的语言代码
            selected_idx = lang_labels.index(lang_label)
            if 0 <= selected_idx < len(languages):
                lang_code = languages[selected_idx]
                selected_ocr_lang = lang_code
                config["ocr_lang"] = lang_code
                save_config()
                # 重置PaddleOCR实例以使用新语言
                reset_paddleocr_instance()
                target = SUPPORTED_LANGUAGES[lang_code]['target_lang']
                print(f"[成功] 已切换: 识别{SUPPORTED_LANGUAGES[lang_code]['label']}, 翻译成{target}")
                signal_bus.update_tips.emit(f"[成功] 识别{SUPPORTED_LANGUAGES[lang_code]['label']} -> 翻译成{target}")

    def _toggle_merge_settings(self):
        if self.merge_settings_widget and self.merge_settings_widget.isVisible():
            self.merge_settings_widget.hide()
        else:
            if self.merge_settings_widget:
                # 摆放到屏幕右下角
                screen = QGuiApplication.primaryScreen()
                geo = screen.availableGeometry()
                self.merge_settings_widget.move(geo.right() - 240, geo.bottom() - 130)
                self.merge_settings_widget.show()

    def _toggle_ocr_precision_settings(self):
        """切换OCR精度设置窗口的显示/隐藏"""
        if self.ocr_precision_widget and self.ocr_precision_widget.isVisible():
            self.ocr_precision_widget.hide()
        else:
            if self.ocr_precision_widget:
                # 摆放到屏幕右下角（在合并设置窗口上方）
                screen = QGuiApplication.primaryScreen()
                geo = screen.availableGeometry()
                self.ocr_precision_widget.move(geo.right() - 240, geo.bottom() - 280)
                self.ocr_precision_widget.show()

    def _select_model(self):
        global selected_model
        models = get_available_models()
        if not models:
            QMessageBox.warning(None, "错误", "无法获取模型列表，请确认Ollama正在运行")
            return
        current_idx = models.index(selected_model) if selected_model in models else 0
        model, ok = QInputDialog.getItem(
            None, "选择翻译模型", "可用模型:", models, current_idx, False
        )
        if ok and model:
            selected_model = model
            config["model"] = model
            save_config()
            print(f"[成功] 已切换模型: {model}")
            signal_bus.update_tips.emit(f"[成功] 模型: {model}")

    def notify(self, title: str, message: str):
        self.tray.showMessage(title, message, QSystemTrayIcon.Information, 3000)

    def _open_hotkey_settings(self):
        dialog = HotkeySettingsDialog()
        dialog.exec()

    def _open_auto_translate(self):
        dialog = AutoTranslateDialog()
        dialog.exec()


# ====================== 自动翻译设置界面 ======================
class AutoTranslateDialog(QDialog):
    """自动翻译设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动翻译设置")
        self.setFixedSize(320, 160)
        self.setStyleSheet("""
            QWidget { font-size: 13px; }
            QCheckBox { spacing: 8px; font-size: 14px; font-weight: bold; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QSlider::groove:horizontal { height: 6px; background: #ccc; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4a9eff; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
            QPushButton { padding: 4px 16px; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.auto_check = QCheckBox("启用自动翻译")
        self.auto_check.toggled.connect(self._on_toggled)
        layout.addWidget(self.auto_check)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("检测间隔:"))
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 50)
        self.interval_slider.setValue(10)
        self.interval_label = QLabel("1.0s")
        self.interval_label.setFixedWidth(50)
        self.interval_slider.valueChanged.connect(self._on_interval_changed)
        interval_row.addWidget(self.interval_slider)
        interval_row.addWidget(self.interval_label)
        layout.addLayout(interval_row)

        hint = QLabel("开启后程序将按间隔不断检测屏幕变化，\n检测到变化自动启动翻译覆盖流程。")
        hint.setStyleSheet("color:#888;font-size:11px;")
        layout.addWidget(hint)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_toggled(self, checked: bool):
        ms = self.interval_slider.value() * 100
        signal_bus.auto_mode_changed.emit(checked, ms)

    def _on_interval_changed(self, val: int):
        ms = val * 100
        self.interval_label.setText(f"{val/10:.1f}s")
        if self.auto_check.isChecked():
            signal_bus.auto_mode_changed.emit(True, ms)


# ====================== 快捷键设置界面 ======================
class HotkeySettingsDialog(QDialog):
    """快捷键设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("快捷键设置")
        self.setFixedSize(420, 400)
        self._recording_action = None
        self._hotkey_listener = _global_hotkey_listener
        # 暂存修改（key=action_name, value=keys列表），保存时才写入 hotkey_config
        self._pending = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("点击快捷键文字后按下新组合键")
        title.setStyleSheet("font-size:13px;color:#555;")
        layout.addWidget(title)

        self._rows = {}
        for action_name in ["recognize", "switch_screen", "toggle_auto", "exit_app", "toggle_green_boxes"]:
            info = DEFAULT_HOTKEYS[action_name]
            row = QHBoxLayout()

            label = QLabel(info["label"])
            label.setFixedWidth(80)

            display = QLineEdit()
            display.setReadOnly(True)
            display.setText(format_hotkey_display(action_name))
            display.setAlignment(Qt.AlignCenter)
            display.setStyleSheet("""
                QLineEdit {
                    font-size:14px; font-weight:bold;
                    padding:4px; border:1px solid #ccc; border-radius:4px;
                }
                QLineEdit:hover { border-color: #3498db; background:#f0f8ff; }
                QLineEdit[recording=\"true\"] {
                    border-color: #e67e22; background: #fff3e0;
                }
            """)
            display.mousePressEvent = lambda e, a=action_name, d=display: self._start_recording(a, d)

            row.addWidget(label)
            row.addWidget(display)
            layout.addLayout(row)
            self._rows[action_name] = display

        layout.addStretch()

        # 底部按钮
        btn_row = QHBoxLayout()
        reset_btn = QPushButton("恢复默认")
        reset_btn.setFixedWidth(100)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        save_btn = QPushButton("保存")
        save_btn.setFixedWidth(80)
        save_btn.setStyleSheet("font-weight:bold;")
        save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(save_btn)
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#e67e22;font-size:12px;")
        self._hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._hint)

    def _start_recording(self, action_name: str, display: QLineEdit):
        self._recording_action = action_name
        self._hint.setText("请按下新的快捷键组合（Ctrl/Alt/Shift+数字/字母）...")
        for d in self._rows.values():
            d.setProperty("recording", False)
            d.setStyleSheet("""
                QLineEdit {
                    font-size:14px; font-weight:bold; padding:4px;
                    border:1px solid #ccc; border-radius:4px;
                }
            """)
        display.setProperty("recording", True)
        display.setText("按下组合键...")
        display.setStyleSheet("""
            QLineEdit {
                font-size:14px; font-weight:bold; padding:4px;
                border:2px solid #e67e22; border-radius:4px;
                background:#fff3e0; color:#e67e22;
            }
        """)
        if self._hotkey_listener and hasattr(self._hotkey_listener, 'start_recording'):
            self._hotkey_listener.start_recording(self._on_record_key)

    def _on_record_key(self, key):
        if self._recording_action is None:
            return
        k = self._normalize_key_local(key)
        if self._is_modifier_local(k):
            return
        recorder = self._hotkey_listener
        if recorder is None or not hasattr(recorder, 'get_pressed_modifiers'):
            return
        pressed_keys = recorder.get_pressed_modifiers()
        pressed_keys.append(k)
        # 存入暂存区
        self._pending[self._recording_action] = pressed_keys
        # 更新显示
        display = self._rows.get(self._recording_action)
        if display:
            display.setText(format_hotkey_display(keys=pressed_keys))
            display.setProperty("recording", False)
            display.setStyleSheet("""
                 QLineEdit {
                     font-size:14px; font-weight:bold; padding:4px;
                     border:1px solid #27ae60; border-radius:4px; background:#eafaf1; color:#27ae60;
                 }
             """)
        self._recording_action = None
        self._hint.setText("已录制，点击「保存」生效")
        if self._hotkey_listener:
            self._hotkey_listener.stop_recording()
        return True

    def _save_and_close(self):
        """保存所有暂存修改"""
        global hotkey_config
        if self._pending:
            for action, keys in self._pending.items():
                hotkey_config[action] = keys
            save_config()
        self.close()

    def _reset_defaults(self):
        """恢复所有快捷键为默认值"""
        self._pending = {}
        for action_name in self._rows:
            keys = DEFAULT_HOTKEYS[action_name]["keys"]
            self._pending[action_name] = keys
            display = self._rows[action_name]
            display.setText(format_hotkey_display(keys=keys))
        self._hint.setText("已恢复默认，点击「保存」生效")
        self._hint.setStyleSheet("color:#27ae60;font-size:12px;")

    def closeEvent(self, event):
        if self._hotkey_listener:
            self._hotkey_listener.stop_recording()
        super().closeEvent(event)

    @staticmethod
    def _is_modifier_local(key_str: str) -> bool:
        return any(m in key_str for m in ['ctrl', 'alt', 'shift', 'cmd'])

    @staticmethod
    def _normalize_key_local(key) -> str:
        try:
            if hasattr(key, 'name') and key.name is not None:
                return key.name.lower()
            if hasattr(key, 'char') and key.char is not None:
                return key.char
            vk = getattr(key, 'vk', None)
            if vk is not None:
                vk_str = str(vk)
                return chr(vk).lower() if 32 <= vk <= 126 else vk_str
            return str(key)
        except Exception:
            return str(key)


# 全局引用（供快捷键设置界面获取监听器）
_global_hotkey_listener = None


# ====================== 快捷键辅助函数 ======================

def get_hotkey_keys(action_name: str) -> list:
    """获取指定动作的快捷键键列表（从配置或默认）"""
    custom = hotkey_config.get(action_name)
    if custom and isinstance(custom, list) and len(custom) > 0:
        return custom
    return DEFAULT_HOTKEYS[action_name]["keys"]


def format_hotkey_display(action_name: str = None, keys: list = None) -> str:
    """将快捷键键列表格式化为可读字符串"""
    if keys is None and action_name is not None:
        keys = get_hotkey_keys(action_name)
    if not keys:
        return "未设置"
    parts = []
    for k in keys:
        k = str(k).lower()
        name = KEY_NAMES.get(k)
        if name:
            parts.append(name)
        elif len(k) == 1 and k.isprintable():
            parts.append(k.upper())
        elif k.startswith('key.'):
            parts.append(k[4:].upper())
        else:
            parts.append(k.upper())
    return "+".join(parts)


# ====================== 快捷键监听 ======================
class HotkeyListener:

    def __init__(self):
        self.listener = None
        self._pressed = set()
        self._action_map = {}
        self._recording_callback = None
        global _global_hotkey_listener
        _global_hotkey_listener = self

    def start(self):
        if not PYNPUT_AVAILABLE:
            print("[警告] pynput未安装，快捷键不可用")
            return

        self._update_action_map()

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.daemon = True
        self.listener.start()
        a = format_hotkey_display("recognize")
        b = format_hotkey_display("switch_screen")
        c = format_hotkey_display("toggle_auto")
        d = format_hotkey_display("exit_app")
        e = format_hotkey_display("toggle_green_boxes")
        print(f"[成功] 快捷键已启用: {a}识别, {b}切屏, {c}自动翻译, {e}绿色框, {d}退出")

    def stop(self):
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass

    def _update_action_map(self):
        self._action_map = {
            "recognize": signal_bus.run_recognize.emit,
            "switch_screen": signal_bus.switch_screen.emit,
            "toggle_auto": signal_bus.toggle_auto.emit,
            "exit_app": signal_bus.exit_app.emit,
            "toggle_green_boxes": signal_bus.toggle_green_boxes.emit,
        }

    def _normalize_key(self, key) -> str:
        """将 pynput 按键对象转为字符串标识"""
        try:
            if hasattr(key, 'name') and key.name is not None:
                return key.name.lower()
            if hasattr(key, 'char') and key.char is not None:
                return key.char
            vk = getattr(key, 'vk', None)
            if vk is not None:
                vk_str = str(vk)
                return chr(vk).lower() if 32 <= vk <= 126 else vk_str
            return str(key)
        except Exception:
            return str(key)

    def _is_modifier(self, key_str: str) -> bool:
        return any(m in key_str for m in ['ctrl', 'alt', 'shift', 'cmd'])

    def _on_press(self, key):
        k = self._normalize_key(key)
        self._pressed.add(k)

        if self._recording_callback:
            self._recording_callback(key)
            return

        for action_name, emit_fn in self._action_map.items():
            target_keys = set(get_hotkey_keys(action_name))
            if target_keys and target_keys == self._pressed:
                emit_fn()
                break

    def _on_release(self, key):
        k = self._normalize_key(key)
        self._pressed.discard(k)

    def get_pressed_modifiers(self) -> list:
        return [k for k in self._pressed if self._is_modifier(k)]

    def start_recording(self, callback):
        """开始录制快捷键（callback 接收 pynput key 对象）"""
        self._recording_callback = callback
        self._pressed.clear()

    def stop_recording(self):
        """停止录制快捷键"""
        self._recording_callback = None


# ====================== 主应用 ======================
class TransGlassApp:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        load_config()
        print(f"[信息] 当前模型: {selected_model}")
        print(f"[信息] 当前OCR语言: {SUPPORTED_LANGUAGES.get(selected_ocr_lang, {}).get('label', selected_ocr_lang)} ({selected_ocr_lang})")

        # 检查/启动 Ollama
        if not ensure_ollama_running():
            reply = QMessageBox.question(
                None, "Ollama未启动",
                "Ollama服务未启动，是否继续？\n（翻译功能将不可用）"
            )
            if reply != QMessageBox.Yes:
                sys.exit(1)

        # 检查当前选择的模型是否存在
        available_models = get_available_models()
        if available_models and selected_model not in available_models:
            QMessageBox.warning(
                None, "模型不存在",
                f"当前选择的模型 '{selected_model}' 不存在！\n\n"
                f"请通过托盘菜单切换到已有模型。\n"
                f"当前可用模型:\n" + "\n".join(f"  • {m}" for m in available_models)
            )

        self.overlay = OverlayWindow()
        self.merge_settings = MergeSettingsWidget()
        self.ocr_precision_settings = OCRPrecisionSettingsWidget()
        self.tray = SystemTray(self.app, self.merge_settings, self.ocr_precision_settings)
        self.hotkey = HotkeyListener()
        self.recognize_thread: Optional[RecognizeThread] = None
        self._is_translating = False
        self._pending_retranslate = False

        # 自动翻译
        self._auto_timer = QTimer()
        self._auto_timer.setParent(self.app)  # 用 QApplication 做 parent
        self._auto_timer.timeout.connect(self._auto_check)
        self._last_hash = 0
        self._last_auto_img_path = ""
        self._auto_interval = 1000  # 自动翻译检测间隔（毫秒）
        self._auto_mode_active = False  # 自动翻译是否已开启

        signal_bus.update_tips.connect(self.overlay.show_tip)
        signal_bus.run_recognize.connect(self._run_recognize)
        signal_bus.switch_screen.connect(self.overlay.switch_screen)
        signal_bus.run_test.connect(self._run_test_pattern)
        signal_bus.exit_app.connect(self._exit)
        signal_bus.auto_mode_changed.connect(self._on_auto_mode_changed)
        signal_bus.toggle_auto.connect(self._toggle_auto)
        signal_bus.toggle_green_boxes.connect(self._toggle_green_boxes)

        self.hotkey.start()
        self.overlay.show_tip("[成功] TransGlass已启动")
        rec_hk = format_hotkey_display("recognize")
        sw_hk = format_hotkey_display("switch_screen")
        auto_hk = format_hotkey_display("toggle_auto")
        exit_hk = format_hotkey_display("exit_app")
        gbox_hk = format_hotkey_display("toggle_green_boxes")
        self.tray.notify("TransGlass", f"已启动\n{rec_hk}: 识别翻译\n{sw_hk}: 切换屏幕\n{auto_hk}: 自动翻译\n{gbox_hk}: 原始识别框\n{exit_hk}: 退出")

    def _run_recognize(self):
        if self.recognize_thread and self.recognize_thread.isRunning():
            self._pending_retranslate = True
            self.overlay.show_tip("[信息] 翻译中，完成后将自动重试...")
            return
        self._is_translating = True
        self._pending_retranslate = False
        self.overlay.clear_translations()
        self.recognize_thread = RecognizeThread(
            self.overlay.current_screen_idx,
            selected_ocr_lang,
            selected_merge_rule
        )
        self.recognize_thread.finished.connect(self._on_finished)
        self.recognize_thread.error.connect(self._on_error)
        self.recognize_thread.start()
        QTimer.singleShot(600, lambda: self.overlay.show_tip("[信息] 正在截取屏幕..."))

    def _on_finished(self, data: list, original_blocks: list, scale_factor: float, img_path: str, text_blocks: list):
        self._is_translating = False
        print(f"[成功] 识别完成，共 {len(data)} 处翻译")

        if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
            translations = [item['text'] for item in data]
            create_translated_overlay(
                image_path=img_path,
                text_blocks=text_blocks,
                translations=translations
            )
            self.overlay.show_tip(f"[测试] 效果图已生成")
            return

        # 无翻译结果（单个字母跳过等）
        if not data:
            print("[自动跳过] 无需显示翻译覆盖")
            if self._auto_mode_active:
                self._update_auto_hash()
                self._auto_timer.start(self._auto_interval)
            return

        self.overlay.set_original_text_blocks(original_blocks, scale_factor)
        self.overlay.set_translations(data)

        # 自动翻译模式：更新hash（带覆盖层），启动定时器继续检测
        if self._auto_mode_active:
            self._update_auto_hash()
            self._auto_timer.start(self._auto_interval)

        if self._pending_retranslate:
            self._pending_retranslate = False
            QTimer.singleShot(200, self._run_recognize)

    def _update_auto_hash(self):
        """翻译完成后更新 hash，使覆盖层变化不干扰下次检测"""
        # 强制处理Qt事件队列，确保覆盖层已完全绘制到屏幕上
        self.app.processEvents()
        self._capture_auto_hash()

    def _capture_auto_hash(self):
        try:
            idx = self.overlay.current_screen_idx
            screens = get_screens_info()
            if idx >= len(screens): return
            s = screens[idx]
            temp = os.path.join(os.path.dirname(CONFIG_PATH), "transglass_auto_temp.png")
            with mss.mss() as shooter:
                shot = shooter.grab(s)
                Image.frombytes('RGB', shot.size, shot.rgb).save(temp, "PNG")
            self._last_hash = compute_dhash(temp)
        except Exception:
            pass
        
    def _run_test_pattern(self):
        """运行坐标测试模式 - 在屏幕四角和中心绘制测试点"""
        print("[测试] 运行坐标测试模式...")
        import random
        
        # 获取当前屏幕信息
        screens = get_screens_info()
        current_idx = self.overlay.current_screen_idx
        if not screens or current_idx >= len(screens):
            self.overlay.show_tip("[错误] 无法获取屏幕信息")
            return
            
        s = screens[current_idx]
        print(f"[测试] 当前屏幕: {s['width']}x{s['height']} @ ({s['left']}, {s['top']})")
        
        # 获取当前屏幕的缩放因子，将物理坐标转换为逻辑坐标
        scale = self.overlay.screen_scale if self.overlay.screen_scale > 0 else 1.0
        # 逻辑尺寸
        lw = int(s['width'] / scale)
        lh = int(s['height'] / scale)
        
        # 生成测试点：四角 + 中心 + 随机点（使用逻辑像素坐标）
        test_points = [
            {'x': 50,          'y': 50,       'width': 100, 'height': 30, 'text': '左上角'},
            {'x': lw - 150,    'y': 50,       'width': 100, 'height': 30, 'text': '右上角'},
            {'x': 50,          'y': lh - 80,  'width': 100, 'height': 30, 'text': '左下角'},
            {'x': lw - 150,    'y': lh - 80,  'width': 100, 'height': 30, 'text': '右下角'},
            {'x': lw//2 - 50,  'y': lh//2 - 15, 'width': 100, 'height': 30, 'text': '屏幕中心'},
        ]
        
        # 添加随机点
        for i in range(3):
            rx = random.randint(100, lw - 200)
            ry = random.randint(100, lh - 200)
            test_points.append({
                'x': rx, 'y': ry, 'width': 120, 'height': 30,
                'text': f'随机点{i+1}'
            })
        
        self.overlay.set_translations(test_points)
        self.overlay.show_tip(f"[测试] 已绘制 {len(test_points)} 个测试点")
        print(f"[测试] 请检查测试点是否出现在屏幕对应位置")

    def _on_error(self, error: str):
        self._is_translating = False
        print(f"[错误] {error}")
        self.overlay.show_tip(f"[失败] {error}")
        if self._pending_retranslate:
            self._pending_retranslate = False
            QTimer.singleShot(500, self._run_recognize)

    def _auto_check(self):
        """自动检测屏幕变化 -> 触发翻译"""
        if self._is_translating or (self.recognize_thread and self.recognize_thread.isRunning()):
            return
        idx = self.overlay.current_screen_idx
        screens = get_screens_info()
        if idx >= len(screens):
            return
        s = screens[idx]
        temp = os.path.join(os.path.dirname(CONFIG_PATH), "transglass_auto_temp.png")
        with mss.mss() as shooter:
            shot = shooter.grab(s)
            Image.frombytes('RGB', shot.size, shot.rgb).save(temp, "PNG")
        h = compute_dhash(temp)
        if h != self._last_hash:
            self._last_hash = h
            self._auto_timer.stop()  # 识别期间停止计时，完成后由 _on_finished 重启
            self._run_recognize()

    def _on_auto_mode_changed(self, enabled: bool, interval_ms: int):
        if enabled:
            self._auto_interval = interval_ms
            print(f"[自动翻译] 已开启，检测间隔={interval_ms}ms")
        else:
            self._auto_timer.stop()
            print(f"[自动翻译] 已关闭")

    def _toggle_auto(self):
        """快捷键切换自动翻译开关"""
        if self._auto_mode_active:
            # 关闭自动翻译
            self._auto_timer.stop()
            self._auto_mode_active = False
            signal_bus.auto_mode_changed.emit(False, 1000)
            self.overlay.show_tip("[自动翻译] 已关闭")
        else:
            # 开启自动翻译：先截图翻译1次，完成后启动定时器
            self._last_hash = 0
            self._auto_interval = 1000
            self._auto_mode_active = True
            signal_bus.auto_mode_changed.emit(True, 1000)
            self._run_recognize()

    def _toggle_green_boxes(self):
        """切换绿色原始识别框显示"""
        self.overlay.show_original_boxes = not self.overlay.show_original_boxes
        self.overlay.update()
        status = "显示" if self.overlay.show_original_boxes else "隐藏"
        print(f"[信息] 原始识别框: {status}")
        self.overlay.show_tip(f"[原始识别框] {status}")

    def _exit(self):
        print("[信息] 正在退出...")
        self.hotkey.stop()
        if self.recognize_thread and self.recognize_thread.isRunning():
            self.recognize_thread.stop()
            self.recognize_thread.wait(2000)
        self.app.quit()
        # 强制退出进程
        import os
        os._exit(0)

    def run(self) -> int:
        print("\n" + "=" * 60)
        print("TransGlass 已启动（OCR引擎: PaddleOCR CPU）")
        print("=" * 60)
        print("快捷键:")
        rec_hk = format_hotkey_display("recognize")
        sw_hk = format_hotkey_display("switch_screen")
        auto_hk = format_hotkey_display("toggle_auto")
        exit_hk = format_hotkey_display("exit_app")
        gbox_hk = format_hotkey_display("toggle_green_boxes")
        print(f"  {rec_hk} - 识别翻译当前屏幕")
        print(f"  {sw_hk} - 切换屏幕")
        print(f"  {auto_hk} - 自动翻译开关")
        print(f"  {gbox_hk} - 原始识别框开关")
        print(f"  {exit_hk} - 退出程序")
        print("=" * 60 + "\n")
        return self.app.exec()


# ====================== 入口 ======================
if __name__ == "__main__":
    if not PYSIDE6_AVAILABLE:
        print("[错误] PySide6未安装，请运行: pip install PySide6")
        sys.exit(1)
    if not MSS_AVAILABLE:
        print("[错误] mss未安装，请运行: pip install mss")
        sys.exit(1)
    if not PYNPUT_AVAILABLE:
        print("[警告] pynput未安装，快捷键不可用，请运行: pip install pynput")

    app = TransGlassApp()
    sys.exit(app.run())
