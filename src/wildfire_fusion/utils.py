from __future__ import annotations
import cv2
import numpy as np

def EMA(prev, cur, alpha=0.2):
    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur

def to_gray(img):
    if img is None:
        raise ValueError("Empty image")
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
