# SuperRes-Pano360

PanoVision360 is a Python-based project that facilitates the creation of 360-degree panoramas from videos. It achieves this by splitting the video into shots, enhancing the image quality using SRCNN (Super-Resolution Convolutional Neural Network), and then stitching the enhanced images together.

## Features

- **Shot Extraction**: The project extracts frames from a video at specified intervals.
- **Image Enhancement**: Optionally, the extracted frames can be enhanced using SRCNN to improve image quality.
- **Panorama Stitching**: Finally, the enhanced frames are stitched together to create a seamless 360-degree panorama.

## Usage

**Test Environment**: The project was tested on an Apple MacBook M2 (MacOS), Windows (You Should change directory on source code) 

### Prerequisites

- Python 3.9
- OpenCV
- TensorFlow
- Pre-train Model Weight (Link)[https://github.com/g2zac/pre-trained-weights-for-the-SRCNN]

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/WellshCorgi/SuperRes-Pano360.git
    cd SuperRes-Pano360
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Instructions

1. Place your video file (e.g., `input.mp4`) in the project directory.
2. Run the script `mp4_to_pic.py` to extract frames from the video and optionally enhance them:

    ```bash
    python mp4_to_pic.py -i 1 -f 60
    ```

    - Use the `-i` flag with `1` to enable image enhancement using SRCNN. (if you don't want to use it  `0`  )
    - Adjust the frame interval using the `-f` flag as needed (default is `60`).
3. Once frames are extracted and enhanced (if selected), run the script `stitching_image_2_pano.py` to stitch the frames into a panorama:

    ```bash
    python stitching_image_2_pano.py
    ```

4. The stitched panorama will be saved as `stitched_result.jpg` in the project directory.

## Credits

This project utilizes the SRCNN model for image enhancement. The SRCNN model architecture and weights are based on the work by Dong et al. [1].

## References

[1] Dong, C., Loy, C. C., He, K., & Tang, X. (2016). Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2), 295-307.

### Additional Information
**Note**: The resulting image `stitched_result_rooftop.jpg` was captured with a Canon mirrorless camera in 4K resolution and stitched using SupreRes-Pano360. The copyright for the photograph belongs to the author.
