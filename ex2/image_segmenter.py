import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches


class ImageSegmenter:
    def __init__(self, mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode= mode

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
        
    def extract_features_(self, sample_dd):
        """ Extract features from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape
        
        #
        # TO IMPLEMENT
        #

        # For now this only extracts intensities
        return img
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        
        #
        # TO IMPLEMENT
        #

        # For now return scribble
        return self.segment_image_dummy(sample_dd)

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
