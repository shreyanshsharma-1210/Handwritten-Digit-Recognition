import sys
import os
import numpy as np
import pytest

# Adjust path to find src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import normalize_images, reshape_for_cnn

def test_normalize_images():
    """Test that image pixels are correctly scaled to [0, 1]."""
    test_img = np.array([[0, 255], [127, 64]], dtype='uint8')
    normalized = normalize_images(test_img)
    
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    assert normalized.dtype == 'float32'
    assert normalized[0, 1] == 1.0

def test_reshape_for_cnn():
    """Test that a channel dimension is correctly added for CNN input."""
    test_imgs = np.zeros((10, 28, 28))
    reshaped = reshape_for_cnn(test_imgs)
    
    assert reshaped.shape == (10, 28, 28, 1)

def test_reshape_idempotent():
    """Test that reshaping an already reshaped image doesn't add extra dimensions."""
    test_imgs = np.zeros((10, 28, 28, 1))
    reshaped = reshape_for_cnn(test_imgs)
    
    assert reshaped.shape == (10, 28, 28, 1)
