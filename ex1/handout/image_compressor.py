import numpy as np


class ImageCompressor:
    """
    Image compression using PCA (Principal Component Analysis).
    
    - Codebook stores: mean image + K principal components + color palette
    - Compressed code: K PCA coefficients (very small!)
    - Reconstruction: PCA inverse + quantize to palette colors
    """
    
    def __init__(self, n_components=23):
        self.n_components = n_components
        self.dtype = np.float16
        self.mean = None
        self.components = None
        
        # Color palette for final quantization (denoising)
        self.palette = np.array([
            [255, 255, 255],  # 0: White (background)
            [0, 0, 0],        # 1: Black (grid lines)
            [212, 50, 48],    # 2: Red #D43230 (X marks)
            [61, 137, 68],    # 3: Green #3D8944 (O marks)
            [128, 128, 128],  # 4: Gray #808080
            [237, 172, 171],  # 5: Pink #EDACAB
            [142, 186, 146],  # 6: Light Green #8EBA92
        ], dtype=np.uint8)
        
        self.H = 96
        self.W = 96
        self.C = 3
        self.D = self.H * self.W * self.C  # 27648

    def get_codebook(self):
        """
        Codebook structure:
        - Row 0: mean image (flattened, D values)
        - Rows 1 to K: principal components (each D values)
        - Row K+1: palette colors (21 values, rest zeros)
        """
        # Stack mean and components
        pca_data = np.vstack([self.mean.reshape(1, -1), self.components])
        
        # Add palette as last row (padded with zeros)
        palette_row = np.zeros((1, self.D), dtype=np.float32)
        palette_row[0, :21] = self.palette.flatten().astype(np.float32)
        
        codebook = np.vstack([pca_data, palette_row])
        return codebook.astype(self.dtype)

    def train(self, train_images):
        """
        Learn PCA from training images.
        """
        N = len(train_images)
        
        # Stack all images as rows: (N, D)
        X = np.zeros((N, self.D), dtype=np.float32)
        for i, img in enumerate(train_images):
            X[i] = img.flatten().astype(np.float32)
        
        # Compute mean
        self.mean = np.mean(X, axis=0)
        
        # Center data
        X_centered = X - self.mean
        
        # SVD for PCA (more numerically stable than covariance)
        # X_centered = U @ S @ Vt
        # Principal components are rows of Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top K components
        self.components = Vt[:self.n_components]  # Shape: (K, D)

    def compress(self, test_image):
        """
        Compress by projecting onto principal components.
        Returns K coefficients.
        """
        img = test_image.flatten().astype(np.float32)
        centered = img - self.mean
        
        # Project onto components: coeffs[k] = components[k] · centered
        coeffs = self.components @ centered  # Shape: (K,)
        
        return coeffs.astype(self.dtype)


class ImageReconstructor:
    """Reconstruct images from PCA coefficients using the codebook."""
    
    def __init__(self, codebook):
        """
        Parse codebook:
        - Row 0: mean
        - Rows 1 to -1: principal components  
        - Last row first 21 values: palette
        """
        codebook = codebook.astype(np.float32)
        
        # Extract palette from last row
        palette_flat = codebook[-1, :21]
        self.palette = palette_flat.reshape(7, 3).astype(np.float32)
        
        # Mean and components
        self.mean = codebook[0]
        self.components = codebook[1:-1]  # All rows except first and last
        
        self.H = 96
        self.W = 96

    def reconstruct(self, test_code):
        """
        Reconstruct from PCA coefficients:
        1. Linear combination of components + mean
        2. Quantize to nearest palette color
        """
        coeffs = test_code.astype(np.float32)
        
        # PCA reconstruction: mean + sum(coeff_k * component_k)
        reconstructed = self.mean + coeffs @ self.components
        
        # Reshape to (N_pixels, 3)
        reconstructed = reconstructed.reshape(-1, 3)
        
        # Quantize to nearest palette color
        diff = reconstructed[:, np.newaxis, :] - self.palette[np.newaxis, :, :]
        distances = np.sum(diff * diff, axis=2)
        indices = np.argmin(distances, axis=1)
        
        # Lookup colors
        result = self.palette[indices].astype(np.uint8)
        
        return result.reshape(self.H, self.W, 3)
