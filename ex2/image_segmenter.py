import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches


class ImageSegmenter:
    def __init__(self, mode='kmeans', k_fg=8, k_bg=15, use_position=True, position_weight=0.5,
                 use_hsv=True, use_patches=False, patch_size=3, use_morphology=True):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode = mode
        self.k_fg = k_fg  # Number of clusters for foreground
        self.k_bg = k_bg  # Number of clusters for background
        self.use_position = use_position  # Whether to use position features
        self.position_weight = position_weight  # Weight for position features
        self.use_hsv = use_hsv  # Whether to use HSV color space
        self.use_patches = use_patches  # Whether to use patch features
        self.patch_size = patch_size  # Patch size for neighborhood features
        self.use_morphology = use_morphology  # Whether to apply morphological post-processing

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
    
    def rgb_to_hsv(self, rgb):
        """ Convert RGB image to HSV color space """
        rgb = rgb.astype(np.float64) / 255.0
        H, W, C = rgb.shape
        hsv = np.zeros_like(rgb)
        
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        diff = maxc - minc
        
        # Value
        hsv[:,:,2] = maxc
        
        # Saturation
        mask = maxc != 0
        hsv[:,:,1][mask] = diff[mask] / maxc[mask]
        
        # Hue
        mask = diff != 0
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc[mask] - r[mask]) / diff[mask]
        gc[mask] = (maxc[mask] - g[mask]) / diff[mask]
        bc[mask] = (maxc[mask] - b[mask]) / diff[mask]
        
        h = np.zeros_like(r)
        mask_r = (maxc == r) & (diff != 0)
        mask_g = (maxc == g) & (diff != 0)
        mask_b = (maxc == b) & (diff != 0)
        
        h[mask_r] = bc[mask_r] - gc[mask_r]
        h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
        h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
        
        hsv[:,:,0] = (h / 6.0) % 1.0
        
        return hsv
        
    def extract_features_(self, sample_dd):
        """ Extract features from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape
        
        # Normalize RGB values to [0, 1]
        rgb_features = img.astype(np.float64) / 255.0
        
        feature_list = [rgb_features]
        
        # Add HSV features
        if self.use_hsv:
            hsv_features = self.rgb_to_hsv(img)
            feature_list.append(hsv_features)
        
        # Add position features
        if self.use_position:
            # Create position grids normalized to [0, 1]
            y_coords = np.arange(H).reshape(-1, 1) / H
            x_coords = np.arange(W).reshape(1, -1) / W
            
            # Broadcast to full image size
            y_grid = np.tile(y_coords, (1, W)).reshape(H, W, 1)
            x_grid = np.tile(x_coords, (H, 1)).reshape(H, W, 1)
            
            # Concatenate position features with weight
            position_features = np.concatenate([y_grid, x_grid], axis=2) * self.position_weight
            feature_list.append(position_features)
        
        # Combine all features
        features = np.concatenate(feature_list, axis=2)
        
        return features
    
    def apply_morphology(self, mask):
        """ Apply morphological operations to clean up the mask using numpy only """
        H, W = mask.shape
        
        def dilate(m, iterations=1):
            """ Dilate mask using 3x3 cross structuring element """
            result = m.copy()
            for _ in range(iterations):
                padded = np.pad(result, 1, mode='constant', constant_values=False)
                dilated = (padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] | 
                          padded[1:-1, :-2] | padded[1:-1, 2:])
                result = dilated
            return result
        
        def erode(m, iterations=1):
            """ Erode mask using 3x3 cross structuring element """
            result = m.copy()
            for _ in range(iterations):
                padded = np.pad(result, 1, mode='constant', constant_values=False)
                eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & 
                         padded[1:-1, :-2] & padded[1:-1, 2:])
                result = eroded
            return result
        
        # Closing (dilate then erode) - fills small holes
        mask = dilate(mask, iterations=2)
        mask = erode(mask, iterations=2)
        
        # Opening (erode then dilate) - removes small noise
        mask = erode(mask, iterations=1)
        mask = dilate(mask, iterations=1)
        
        return mask
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        n_features = features.shape[2]
        
        # Get foreground and background scribble masks
        fg_mask = sample_dd['scribble_fg'] == 255
        bg_mask = sample_dd['scribble_bg'] == 255
        
        # Extract features for foreground and background pixels
        fg_pixels = features[fg_mask].reshape(-1, n_features)
        bg_pixels = features[bg_mask].reshape(-1, n_features)
        
        # Fit k-means on foreground pixels
        fg_centroids = kmeans_fit(fg_pixels, self.k_fg, self.rng)
        
        # Fit k-means on background pixels
        bg_centroids = kmeans_fit(bg_pixels, self.k_bg, self.rng)
        
        # Combine centroids: FG centroids labeled 1 (True), BG centroids labeled 0 (False)
        all_centroids = np.vstack([fg_centroids, bg_centroids])
        centroid_labels = np.array([1] * self.k_fg + [0] * self.k_bg)
        
        # Reshape features for kNN: (H*W, n_features)
        all_pixels = features.reshape(-1, n_features)
        
        # Predict labels using kNN (k=1)
        predicted_labels = kNN(all_centroids, centroid_labels, all_pixels, k=1)
        
        # Reshape predictions back to image shape
        mask_pred = predicted_labels.reshape(H, W).astype(bool)
        
        # Apply morphological post-processing to clean up the mask
        if self.use_morphology:
            mask_pred = self.apply_morphology(mask_pred)
        
        return mask_pred

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
