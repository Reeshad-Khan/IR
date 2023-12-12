## Image Search System Readme

### Overview
This repository contains code for an image search system based on co-occurrence features and the Structural Similarity Index (SSIM). The system utilizes a pre-trained VGG16 model on ImageNet for feature extraction.

### Prerequisites
1. Python 3
2. Flask (`pip install flask`)
3. PyTorch (`pip install torch`)
4. OpenCV (`pip install opencv-python`)
5. NumPy (`pip install numpy`)
6. SciKit-Image (`pip install scikit-image`)

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Reeshad-Khan/IR.git 
   cd IR
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000/`.

5. Choose an image from the `test_images` folder or use your own.

6. Click the "Upload" button and wait for the system to process the image.

7. After processing, the uploaded image and the most similar image will be displayed.

### Folder Structure
- `uploads`: Temporary folder for storing uploaded images.
- `most_similar`: Folder containing the most similar images found during searches.

### Additional Notes
- The system is based on perceptual similarity using SSIM and may not perform well with highly stylized or abstract images.
- Ensure a stable internet connection for the initial loading of the VGG16 model.

Feel free to explore and customize the code for your specific use case!
