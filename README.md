# SIFT Feature Detection Implementation

This repository contains a PyTorch implementation of the Scale-Invariant Feature Transform (SIFT) algorithm for detecting keypoints and extracting feature descriptors from images. The implementation includes various image transformations and visualizations.

## Features

- Scale-space generation using Gaussian blurring
- Keypoint detection through extrema finding
- Orientation assignment for keypoints
- SIFT descriptor extraction
- Principal Component Analysis (PCA) for descriptor dimension reduction
- Image transformation operations:
  - Scaling (resizing)
  - Rotation
  - Gaussian blurring
- Visualization tools for keypoints

## Requirements

```
torch
torchvision
PIL
matplotlib
numpy
scikit-learn
```

## Usage

1. Load and preprocess images:
```python
image_tensor = load_image(image_path)
```

2. Generate scale space and detect keypoints:
```python
scale_space = generate_scale_space(image_tensor, num_scales=5)
keypoints = find_extrema(scale_space)
```

3. Compute gradients and assign orientations:
```python
magnitude, orientation = compute_gradients(image_tensor)
oriented_keypoints = assign_orientation(keypoints, magnitude, orientation)
```

4. Extract SIFT descriptors:
```python
descriptors = extract_sift_descriptors(oriented_keypoints, magnitude, orientation)
```

5. Visualize keypoints:
```python
plot_keypoints(image_tensor, keypoints)
```

## Image Transformations

### Scaling
```python
scaled_image = scale_image_tensor(image_tensor, scale_factor=0.5)
```

### Rotation
```python
rotated_image = rotate_image_tensor(image_tensor, angle=90)
```

### Blurring
```python
blurred_image = apply_gaussian_blur(image_tensor, kernel_size=9, sigma=4.0)
```

## Feature Analysis

The implementation includes PCA for reducing the dimensionality of SIFT descriptors:

```python
pca = PCA(n_components=36)
reduced_descriptors = pca.fit_transform(descriptors)
```

## Visualization

The repository includes various visualization functions for displaying:
- Original and transformed images
- Detected keypoints
- Filtered keypoints within specific image regions

## Note

This implementation is designed for educational purposes and demonstrates the core concepts of the SIFT algorithm. For production use, consider using optimized implementations like OpenCV's SIFT implementation.
