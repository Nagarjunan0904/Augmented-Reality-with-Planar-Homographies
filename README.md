# Augmented Reality with Planar Homographies

This project implements an **Augmented Reality (AR) application** using **planar homographies**.  
It combines feature detection & matching, robust homography estimation, AR overlays, and panorama stitching.

---

## Project Overview

The core objectives are:
- Detecting features in images using **FAST + BRIEF**
- Matching descriptors efficiently with **Hamming distance**
- Estimating homographies (with normalization & RANSAC)
- Warping and compositing images (including videos) onto target surfaces
- Creating a simple **panorama** by stitching images

---

## Repository Structure

.
├── ar.py              # Augmented Reality on video (overlay AR source onto book.mov)
├── briefRotTest.py    # Test BRIEF descriptor robustness to rotation
├── HarryPotterize.py  # Replace cv_cover with hp_cover on desk image
├── helper.py          # BRIEF, matching, plotting, and corner detection utilities
├── loadSaveVid.py     # Video loading and saving utilities
├── matchPics.py       # Feature matching functions (with caching + visualization)
├── panaroma.py        # Image stitching to create panoramas
├── planarH.py         # Homography computation & RANSAC
├── q3_4.py            # Basic feature matching test
└── results/           # Output results (matches, AR videos, panoramas)


---

##  Implementation Highlights

- **Feature Matching**: FAST + BRIEF for efficient corner detection and descriptors.
- **Rotation Test**: Evaluated BRIEF performance across different rotations, visualized with histograms.
- **Homography Estimation**: Implemented direct, normalized, and RANSAC-based approaches.
- **Harry Potterization**: Warped `hp_cover.jpg` onto `cv_desk.png`.
- **Augmented Reality**: Overlayed frames from `ar_source.mov` onto `book.mov`, with homography smoothing for stability.
- **Panorama**: Stitched `pano_left.jpg` and `pano_right.jpg` into a seamless panorama.

---

##  Running the Code

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install numpy opencv-python matplotlib scikit-image scipy

## Scripts

# Feature Matching Test
python q3_4.py

# Rotation Test
python briefRotTest.py

# Harry Potterization (replace cv_cover with hp_cover)
python HarryPotterize.py

# Augmented Reality (Video Overlay)
python ar.py

# Panorama Stitching
python panaroma.py

## Results

- Feature Matching: BRIEF matches decrease with rotation, confirming lack of rotational invariance.
- Homography Estimation: RANSAC improves robustness by filtering outliers.
- Augmented Reality: Achieved smooth and stable video overlays with homography smoothing.
- Panorama: Generated seamless panoramas through feature matching and blending.

##  Acknowledgements

```text
Author     : Nagarjunan Saravanan
Libraries  : OpenCV, NumPy, SciPy, Matplotlib, scikit-image
