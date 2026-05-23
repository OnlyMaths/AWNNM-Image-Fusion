import torch
import torch.nn.functional as F

def adaptive_wnnm_fusion(I1: torch.Tensor, I2: torch.Tensor, 
                        lambda_reg: float = 0.055, 
                        max_iter: int = 30, 
                        rank: int = 80) -> torch.Tensor:
    """
    Adaptive Weighted Nuclear Norm Minimization (AWNNM) for Grayscale Image Fusion.
    
    Args:
        I1, I2: Input grayscale images as tensors of shape (H, W) or (1, H, W), range [0, 1]
        lambda_reg: Regularization parameter (default: 0.055)
        max_iter: Maximum number of iterations (default: 30)
        rank: Target rank for randomized SVD (default: 80)
    
    Returns:
        Fused image as tensor of shape (H, W)
    """
    
    # Ensure inputs are 2D and in [0,1]
    if I1.dim() == 3:
        I1 = I1.squeeze(0)
    if I2.dim() == 3:
        I2 = I2.squeeze(0)
    
    device = I1.device
    I1 = I1.to(device, dtype=torch.float32)
    I2 = I2.to(device, dtype=torch.float32)
    
    # Step 1: Compute average image
    I_bar = (I1 + I2) / 2
    
    # Initialize
    I = I_bar.clone()
    Y = I_bar.clone()
    t = 1.0
    
    epsilon = 1e-6
    
    for k in range(max_iter):
        # FISTA extrapolation
        t_next = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        X = Y + ((t - 1) / t_next) * (Y - I)
        
        # Randomized SVD
        U, S, V = torch.linalg.svd_lowrank(X.unsqueeze(0), q=rank, niter=2)
        S = S.squeeze(0)  # singular values
        
        # Compute adaptive weights
        weights = 1.0 / (S + epsilon)
        
        # Singular value shrinkage
        S_shrunk = torch.maximum(S - lambda_reg * weights, torch.zeros_like(S))
        
        # Reconstruct low-rank approximation
        Z = torch.mm(U.squeeze(0) * S_shrunk.unsqueeze(0), V.squeeze(0))
        
        # Proximal update with fidelity term
        I_new = Z + (1 - lambda_reg) * (I_bar - Z)
        
        # FISTA momentum update
        Y = I_new + ((t - 1) / t_next) * (I_new - I)
        t = t_next
        I = I_new
    
    return I.clamp(0, 1)


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Example: Load two grayscale images
    # I1 = torch.from_numpy(image1).float() / 255.0
    # I2 = torch.from_numpy(image2).float() / 255.0
    
    # fused = adaptive_wnnm_fusion(I1, I2)
    print("Adaptive WNNM Fusion function ready!")
