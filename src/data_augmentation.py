import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter

def rotate_image(image, angle):
    """
    Rotates an image by a specified angle.

    Args:
        image (numpy.ndarray): Input image (28, 28, 1).
        angle (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: Rotated image.
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result.reshape(image.shape)

def shift_image(image, dx, dy):
    """
    Shifts an image horizontally and vertically.

    Args:
        image (numpy.ndarray): Input image (28, 28, 1).
        dx (float): Shift amount in x-direction.
        dy (float): Shift amount in y-direction.

    Returns:
        numpy.ndarray: Shifted image.
    """
    transform_mat = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv2.warpAffine(image, transform_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result.reshape(image.shape)

def zoom_image(image, zoom_factor):
    """
    Zooms into or out of an image.

    Args:
        image (numpy.ndarray): Input image (28, 28, 1).
        zoom_factor (float): Factor to zoom by (e.g., 1.1 for 110% zoom).

    Returns:
        numpy.ndarray: Zoomed image.
    """
    h, w = image.shape[:2]
    zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    zh, zw = zoomed.shape[:2]
    
    if zoom_factor < 1.0:
        pad_h = (h - zh) // 2
        pad_w = (w - zw) // 2
        result = cv2.copyMakeBorder(zoomed, pad_h, h - zh - pad_h, pad_w, w - zw - pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        crop_h = (zh - h) // 2
        crop_w = (zw - w) // 2
        result = zoomed[crop_h:crop_h+h, crop_w:crop_w+w]
        
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return result.reshape(image.shape)

def elastic_deformation(image, alpha=36, sigma=4, random_state=None):
    """
    Applies elastic deformation to an image as described in [Simard2003].

    Args:
        image (numpy.ndarray): Input image (28, 28, 1).
        alpha (float): Scaling factor for the displacement field.
        sigma (float): Standard deviation of the Gaussian filter.
        random_state (numpy.random.RandomState, optional): Random state for reproducible results.

    Returns:
        numpy.ndarray: Elastically deformed image.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image.squeeze(), indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def augment_dataset(images, labels, augmentation_factor=1):
    """
    Augments a dataset by applying random transformations.

    Args:
        images (numpy.ndarray): Collection of images to augment.
        labels (numpy.ndarray): Corresponding labels.
        augmentation_factor (int): How many augmented copies to create per original image.

    Returns:
        tuple: (final_images, final_labels) concatenated with original data and shuffled.
    """
    augmented_images = []
    augmented_labels = []
    
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        
        for _ in range(augmentation_factor):
            aug_type = np.random.choice(['rotate', 'shift', 'zoom', 'elastic'])
            if aug_type == 'rotate':
                angle = np.random.uniform(-15, 15)
                aug_img = rotate_image(img, angle)
            elif aug_type == 'shift':
                dx, dy = np.random.uniform(-2, 2, 2)
                aug_img = shift_image(img, dx, dy)
            elif aug_type == 'zoom':
                zoom_factor = np.random.uniform(0.9, 1.1)
                aug_img = zoom_image(img, zoom_factor)
            else:
                aug_img = elastic_deformation(img)
                
            augmented_images.append(aug_img)
            augmented_labels.append(label)
            
    final_images = np.concatenate((images, np.array(augmented_images)), axis=0)
    final_labels = np.concatenate((labels, np.array(augmented_labels)), axis=0)
    
    idx = np.random.permutation(len(final_images))
    return final_images[idx], final_labels[idx]

def visualize_augmentation(original):
    """
    Returns a dictionary of augmented versions of a single image for plotting.

    Args:
        original (numpy.ndarray): The base image to transform.

    Returns:
        dict: Labels mapping to augmented images.
    """
    return {
        "Original": original,
        "Rotated (15 deg)": rotate_image(original, 15),
        "Rotated (-15 deg)": rotate_image(original, -15),
        "Shifted (+2, +2)": shift_image(original, 2, 2),
        "Zoomed (1.1x)": zoom_image(original, 1.1),
        "Zoomed (0.9x)": zoom_image(original, 0.9),
        "Elastic": elastic_deformation(original)
    }
