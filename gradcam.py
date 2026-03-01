import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAMVisualizer:
    def __init__(self):
        pass
    
    def _create_mock_gradcam(self, image):
        """Create mock heatmap for demonstration"""
        h, w = image.shape[:2]
        # Create a realistic-looking heatmap
        heatmap = np.random.rand(h // 16, w // 16)
        heatmap = cv2.resize(heatmap, (w, h))
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_normalized = np.clip(255 * heatmap, 0, 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        
        # Convert image to BGR if it's RGB
        if image.shape[2] == 3 and np.max(image) <= 1.0:
            image_bgr = (image * 255).astype(np.uint8)
            if image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image.astype(np.uint8)
        
        # Overlay heatmap
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay

# Global visualizer instance
gradcam_visualizer = GradCAMVisualizer()