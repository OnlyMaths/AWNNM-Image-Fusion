"""
Simple script to reproduce fusion results using AWNNM
"""

import torch
from PIL import Image
import numpy as np
from awnnm_fusion import adaptive_wnnm_fusion
import os

def load_image(path):
    """Load image and convert to tensor in [0,1]"""
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img)

def save_image(tensor, path):
    """Save tensor as image"""
    img = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(path)

# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example paths (change these to your image paths)
    ir_path = "data/TNO/IR/1.png"      # Example path
    vi_path = "data/TNO/VI/1.png"
    
    print("Loading images...")
    I1 = load_image(ir_path)
    I2 = load_image(vi_path)
    
    print("Performing fusion with Adaptive WNNM...")
    fused = adaptive_wnnm_fusion(I1, I2, lambda_reg=0.055, max_iter=25)
    
    # Save result
    os.makedirs("results", exist_ok=True)
    save_image(fused, "results/fused_awnnm.png")
    
    print("✅ Fusion completed! Result saved to 'results/fused_awnnm.png'")
