import sys
import os
import numpy as np
import pytest

# Adjust path to find src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentation import rotate_image, shift_image, zoom_image

@pytest.fixture
def sample_image():
    """Returns a dummy image with a central point for tracking transformations."""
    img = np.zeros((28, 28, 1), dtype='float32')
    img[14, 14, 0] = 1.0 # Center pixel
    return img

def test_rotate_image(sample_image):
    """Wait - if we rotate exactly around center, center pixel stays."""
    # Place a pixel at (10, 10)
    img = np.zeros((28, 28, 1), dtype='float32')
    img[10, 10, 0] = 1.0
    
    rotated = rotate_image(img, 90)
    assert rotated.shape == (28, 28, 1)
    # After 90 deg rotation, non-center pixel should move
    assert rotated[10, 10, 0] == 0.0

def test_shift_image(sample_image):
    shifted = shift_image(sample_image, 2, 2)
    assert shifted.shape == (28, 28, 1)
    # Original center was (14, 14), now should be (16, 16)
    assert shifted[14, 14, 0] == 0.0
    assert shifted[16, 16, 0] > 0.0

def test_zoom_image(sample_image):
    # Zoom in
    zoomed_in = zoom_image(sample_image, 1.2)
    # The single center pixel will still be at center
    assert zoomed_in.max() > 0.0
    
    # Zoom out
    zoomed_out = zoom_image(sample_image, 0.5)
    assert zoomed_out.max() > 0.0
