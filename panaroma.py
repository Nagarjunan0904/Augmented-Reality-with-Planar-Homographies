import numpy as np
import cv2
import os
from matchPics import matchPics
from planarH import computeH_ransac

# Paths to the two images
left_img_path = "../data/pano_left.jpg"
right_img_path = "../data/pano_right.jpg"
output_path = "../results/panorama.jpg"

def getHomography(left, right):
    """
    Compute the homography mapping points from the right image to the left image.
    Uses matchPics to detect correspondences, converts keypoints from (row, col) to (x, y),
    and then uses computeH_ransac.
    """
    matches, locs_left, locs_right = matchPics(left, right)
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")
    
    pts_left = locs_left[matches[:, 0], :]    # from left image
    pts_right = locs_right[matches[:, 1], :]    # from right image
    pts_left_xy = pts_left[:, [1, 0]]           # convert to (x, y)
    pts_right_xy = pts_right[:, [1, 0]]         # convert to (x, y)
    
    H, inliers = computeH_ransac(pts_left_xy, pts_right_xy, nSamples=20, threshold=10)
    if H is None:
        raise ValueError("Homography computation failed.")
    # Avoid division by zero if H[2,2] is zero.
    eps = 1e-6
    if np.abs(H[2,2]) < eps:
        H[2,2] = eps
    H = H / H[2,2]
    return H

def warpImages(left, right, H):
    """
    Create a panorama by warping the right image into the left image's coordinate frame.
    The output canvas is sized to fit both images.
    """
    # Get image shapes
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Corners of the right image in homogeneous coordinates.
    corners_right = np.array([
        [0, 0, 1],
        [w_right, 0, 1],
        [w_right, h_right, 1],
        [0, h_right, 1]
    ]).T  # shape 3 x 4
    
    # Warp corners using H (maps right -> left)
    warped_corners = H @ corners_right
    # Avoid division by zero:
    eps = 1e-6
    for j in range(warped_corners.shape[1]):
        if np.abs(warped_corners[2,j]) < eps:
            warped_corners[2,j] = eps
    warped_corners = warped_corners / warped_corners[2, :]
    
    # Combine with left image corners.
    left_corners = np.array([
        [0, 0, 1],
        [w_left, 0, 1],
        [w_left, h_left, 1],
        [0, h_left, 1]
    ]).T
    all_corners = np.hstack((left_corners, warped_corners))
    xs = all_corners[0, :]
    ys = all_corners[1, :]
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    
    # Compute translation to shift panorama into positive coordinates.
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float32)
    
    # Size of panorama
    pano_w = x_max - x_min
    pano_h = y_max - y_min
    
    # Ensure H is float32.
    H = H.astype(np.float32)
    # Warp right image into panorama canvas.
    warped_right = cv2.warpPerspective(right, T @ H, (pano_w, pano_h))
    # Warp left image using only the translation T.
    warped_left = cv2.warpPerspective(left, T, (pano_w, pano_h))
    
    # Blend the two warped images.
    mask_left = (warped_left.sum(axis=2) > 0).astype(np.float32)
    mask_right = (warped_right.sum(axis=2) > 0).astype(np.float32)
    overlap = (mask_left * mask_right) > 0
    
    panorama = warped_left.copy()
    panorama[warped_right.sum(axis=2) > 0] = warped_right[warped_right.sum(axis=2) > 0]
    
    for c in range(3):
        channel_left = warped_left[:,:,c].astype(np.float32)
        channel_right = warped_right[:,:,c].astype(np.float32)
        avg = ((channel_left + channel_right) / 2).astype(np.uint8)
        panorama[:,:,c][overlap] = avg[overlap]
    
    return panorama

def main():
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print("Error: Could not load one or both images. Check file paths.")
        return
    
    H = getHomography(left_img, right_img)
    pano = warpImages(left_img, right_img, H)
    
    cv2.imwrite(output_path, pano)
    cv2.imshow("Panorama", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Panorama saved at {output_path}")

if __name__ == "__main__":
    main()
