import os
import cv2
import numpy as np
from src.data.transforms import load_and_prep_grayscale_to_rgb, prep_frame_grayscale_to_rgb

def test_preprocessing_parity():
    # Create a dummy image
    test_path = "dummy_test_frame.png"
    # Create a random BGR image
    dummy_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(test_path, dummy_bgr)
    
    try:
        # Load via path-based function
        img_rgb_path = load_and_prep_grayscale_to_rgb(test_path)
        
        # Load via in-memory array and array-based function
        frame_bgr = cv2.imread(test_path)
        assert frame_bgr is not None, "Failed to load test image"
        
        img_rgb_mem = prep_frame_grayscale_to_rgb(frame_bgr)
        
        # Assert parity
        np.testing.assert_array_equal(img_rgb_path, img_rgb_mem, err_msg="Preprocessing parity failed!")
        print("Preprocessing parity test passed successfully.")
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

if __name__ == "__main__":
    test_preprocessing_parity()
