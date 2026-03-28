import sys
import os
import pytest
import tensorflow as tf

# Adjust path to find models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_nn import create_simple_nn
from models.cnn_basic import create_basic_cnn
from models.cnn_advanced import create_advanced_cnn

def test_simple_nn_shape():
    """Test that the Simple NN accepts 28x28x1 the output is (1, 10)."""
    model = create_simple_nn()
    input_batch = tf.zeros((1, 28, 28, 1))
    output = model(input_batch)
    
    assert output.shape == (1, 10)

def test_cnn_basic_shape():
    """Test that the Basic CNN output is (1, 10)."""
    model = create_basic_cnn()
    input_batch = tf.zeros((1, 28, 28, 1))
    output = model(input_batch)
    
    assert output.shape == (1, 10)

def test_cnn_advanced_shape():
    """Test that the Advanced CNN output is (1, 10)."""
    model = create_advanced_cnn()
    input_batch = tf.zeros((1, 28, 28, 1))
    output = model(input_batch)
    
    assert output.shape == (1, 10)

def test_model_names():
    """Test that models are correctly named."""
    assert create_simple_nn().name == "simple_nn"
    assert create_basic_cnn().name == "cnn_basic"
    assert create_advanced_cnn().name == "cnn_advanced"
