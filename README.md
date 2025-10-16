# CHROMAGEN: AI IMAGE COLORIZATION WEB APP

**CHROMAGEN** is a user-friendly web application that uses a **deep learning model** to colorize black and white images.  
Built with **Python**, **TensorFlow**, and **Gradio**, it brings grayscale images to life with rich, realistic colors.

---

## How It Works

The application uses a CNN-based colorization pipeline inspired by the *“Colorful Image Colorization”* paper (Zhang et al.):

1. **CIELAB Color Space:**  
   The input image is converted from RGB to *CIELAB*, separating it into:
   - **L channel** → luminance (black-and-white details)  
   - **a, b channels** → color components  

2. **CNN Prediction:**  
   A pre-trained Convolutional Neural Network predicts the *a* and *b* color channels from the *L* channel.

3. **High-Resolution Reconstruction:**  
   The color channels are upscaled and merged with the original *L* channel to preserve detail and clarity.

---

## Tech Stack

- **Language:** Python 3.10 / 3.11 (TensorFlow not supported on Python 3.12+)
- **Frameworks:** TensorFlow (Keras), Gradio
- **Libraries:** NumPy, Scikit-image

---

## Setup Instructions for macOS (M1/M2/M3)

### Clone or Download the Repository
```bash
cd ~/Downloads
unzip ChromaGen.zip     # or use git clone if from GitHub
cd "ChromaGen"          # or your project folder
