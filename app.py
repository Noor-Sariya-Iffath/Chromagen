import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

# --- 1. Define CNN Architecture and Load Weights ---

# Define image dimensions as used by the trained model
img_height = 128
img_width = 128

def get_cnn_model():
    """
    Builds the CNN model architecture.
    """
    model = tf.keras.Sequential([
        # CONV 1
        Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV2
        Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        #CONV3
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV4
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV5 (with dilation)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV6 (with dilation)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV7
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV8
        Conv2DTranspose(filters=256, kernel_size=4, strides=(2,2), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=313, kernel_size=1, strides=(1,1), padding='valid'),

        # OUTPUT
        Conv2D(filters=2, kernel_size=1, padding='valid', dilation_rate=1, strides=(1,1), use_bias=False),
        UpSampling2D(size=4, interpolation='bilinear'),
    ])
    return model

# Build the CNN architecture
model = get_cnn_model()

# Load the pre-trained weights
try:
    model.load_weights('model/colorization_model.h5')
    print("Successfully loaded pre-trained weights into the CNN model.")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Warning: Could not load weights. The model will run with random initializations.")

# --- 2. Define the Image Processing and Colorization Function ---

def colorize_image(input_image):
    """
    This function takes a black and white image, processes it, and returns the colorized version
    at the original input size, preserving original detail for better clarity.
    """
    # --- Step a: Convert original image to Lab and get high-res L channel ---
    # This gives us the high-resolution luminance (detail) channel.
    original_lab = rgb2lab(input_image)
    high_res_l = original_lab[:, :, 0]

    # --- Step b: Pre-process for the model ---
    # Create a low-resolution version of the image for the model input.
    resized_image = resize(input_image, (img_height, img_width), anti_aliasing=True)
    resized_lab = rgb2lab(resized_image)
    
    # The model only needs the L channel as input.
    model_input_l = resized_lab[:, :, 0]
    model_input = model_input_l[np.newaxis, ..., np.newaxis]

    # --- Step c: Model Prediction ---
    # Predict the 'a' and 'b' color channels at low resolution.
    predicted_ab_low_res = model.predict(model_input)[0]

    # --- Step d: Post-process and Upscale Color ---
    # Upscale the predicted 'a' and 'b' channels to the original image size.
    predicted_ab_high_res = resize(predicted_ab_low_res, (original_lab.shape[0], original_lab.shape[1]), anti_aliasing=True)

    # --- Step e: Combine High-Res L with High-Res Color ---
    # Create the final Lab image.
    final_lab_image = np.zeros_like(original_lab)
    final_lab_image[:, :, 0] = high_res_l  # Use original high-res L channel
    final_lab_image[:, :, 1:] = predicted_ab_high_res  # Use upscaled predicted a,b channels

    # --- Step f: Convert to RGB ---
    # Convert the final high-resolution Lab image back to RGB.
    final_rgb_image = lab2rgb(final_lab_image)

    # Convert to 8-bit format for display.
    return (final_rgb_image * 255).astype(np.uint8)

# --- 3. Create the Gradio Web Interface ---

# Custom CSS for a more polished look
css = """
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
}
#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #666;
}
"""

# Use gr.Blocks for more layout control and a better design
with gr.Blocks(theme='gradio/soft', css=css) as demo:
    # Title and Description
    gr.Markdown("<h1 id='title'>ChromaGen: AI Image Colorization</h1>", elem_id="title")
    gr.Markdown("<p id='subtitle'>Upload a black and white image, then click the button to let the AI colorize it.</p>", elem_id="subtitle")

    # Side-by-side layout for input and output
    with gr.Row():
        with gr.Column():
            # Input component
            input_image = gr.Image(type="numpy", label="Upload B&W Image")
            # Example images
            gr.Examples(
                examples=[['static/sample_image_1.jpg']],
                inputs=input_image,
                label="Click an example to try"
            )
        with gr.Column():
            # Output component
            output_image = gr.Image(type="numpy", label="Colorized Result")

    # The button to trigger the colorization
    colorize_button = gr.Button("Colorize Image", variant="primary")
    
    # Connect the button to the colorize_image function
    colorize_button.click(
        fn=colorize_image,
        inputs=input_image,
        outputs=output_image
    )

# --- 4. Launch the Web Application ---

if __name__ == "__main__":
    # The launch() method starts the local web server.
    demo.launch()