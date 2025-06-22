import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

# Import your Generator class from model.py
from model import Generator

# --- Configuration ---
LATENT_DIM = 100  # Should match the latent dimension of your trained model
OUTPUT_DIM = 28 * 28  # Assuming 28x28 pixel grayscale images
MODEL_PATH = "generator_model.pth"

# --- Load Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained generator model."""
    model = Generator(LATENT_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

generator = load_model()

# --- Web App UI ---
st.title("Handwritten Image Digit Generator")

st.write("Generate ")

# User selection for the digit
selected_digit = st.selectbox("Select a Digit", list(range(10)))

# Button to trigger generation
if st.button("Generate Image"):
    #st.subheader(f"Generated Images for Digit: {selected_digit}")

    with st.spinner("Generating images..."):
        # Generate 5 images
        num_images = 5
        # We'll use the selected digit to slightly influence the latent space for variety
        # This is a simple approach; more sophisticated methods exist for conditional generation
        noise = torch.randn(num_images, LATENT_DIM)
        
        # A simple way to condition the generation on the selected digit
        # You might have a more complex conditioning mechanism in your model
        if hasattr(generator, 'label_embedding'): # Check if your model uses label embeddings
             labels = torch.LongTensor([selected_digit] * num_images)
             generated_images = generator(noise, labels)
        else:
             # If not explicitly conditioned, you can still use the digit to seed the noise
             # for some variation.
             noise[:, 0] = selected_digit 
             generated_images = generator(noise)


        # Reshape and process the images for display
        generated_images = generated_images.view(num_images, 1, 28, 28)
        generated_images = (generated_images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]

        # Create a grid of images
        grid = make_grid(generated_images, nrow=5, padding=10, pad_value=1)
        
        # Convert to a displayable format
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_img = Image.fromarray((grid_np * 255).astype(np.uint8))

        st.image(grid_img, use_column_width=True, caption="Generated Digits")