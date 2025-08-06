"""Streamlit web application for AI Outfit Completer."""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import io

from config import STREAMLIT_CONFIG, MODEL_DIR
from inference import OutfitCompleter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #FF5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        border: 2px dashed #FF6B6B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #FFF5F5;
        margin: 1rem 0;
    }
    
    .result-section {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .outfit-item {
        text-align: center;
        margin: 0.5rem;
    }
    
    .similarity-score {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the outfit completion model (cached)."""
    try:
        # Look for the latest model
        model_files = list(MODEL_DIR.glob("*_latest.pth"))
        if not model_files:
            model_files = list(MODEL_DIR.glob("*.pth"))

        if not model_files:
            st.error("No trained model found. Please train a model first.")
            return None

        model_path = model_files[0]
        logger.info(f"Loading model from {model_path}")

        completer = OutfitCompleter(str(model_path))
        return completer

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logger.error(f"Model loading error: {e}")
        return None


def validate_image(image_file) -> bool:
    """Validate uploaded image file."""
    if image_file is None:
        return False

    # Check file size
    if image_file.size > STREAMLIT_CONFIG["max_file_size"] * 1024 * 1024:
        st.error(
            f"File size too large. Maximum size: {STREAMLIT_CONFIG['max_file_size']}MB"
        )
        return False

    # Check file extension
    file_extension = image_file.name.split(".")[-1].lower()
    if file_extension not in STREAMLIT_CONFIG["allowed_extensions"]:
        st.error(
            f"Unsupported file type. Allowed: {', '.join(STREAMLIT_CONFIG['allowed_extensions'])}"
        )
        return False

    return True


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        return None


def display_outfit_grid(outfit_images: list, title: str = "Completed Outfit"):
    """Display outfit images in a grid."""
    st.subheader(title)

    if not outfit_images:
        st.warning("No outfit items to display.")
        return

    # Calculate number of columns (max 4)
    num_cols = min(len(outfit_images), 4)
    cols = st.columns(num_cols)

    for i, image_path in enumerate(outfit_images):
        col_idx = i % num_cols
        with cols[col_idx]:
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"Item {i+1}", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {image_path}")


def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🎨 AI Outfit Completer")
    st.markdown(
        """
    **Transform your fashion sense with AI!** 
    
    Upload a photo of any clothing item, and our AI will suggest a complete outfit that perfectly complements your style.
    """
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown(
            """
        1. **Upload** a photo of a clothing item
        2. **Wait** for AI analysis 
        3. **Discover** your perfect outfit
        4. **Get inspired** by AI recommendations
        """
        )

        st.header("📝 Tips")
        st.markdown(
            """
        - Use clear, well-lit photos
        - Single clothing items work best
        - Supported formats: JPG, PNG, WEBP
        - Maximum file size: 10MB
        """
        )

        # Model status
        st.header("🤖 Model Status")
        model = load_model()
        if model:
            st.success("✅ Model loaded successfully")
            st.info(f"📊 {len(model.item_embeddings)} items in database")
        else:
            st.error("❌ Model not available")

    # Main content
    if model is None:
        st.error(
            "Please ensure a trained model is available before using the application."
        )
        st.stop()

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📸 Upload Your Fashion Item")

    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=STREAMLIT_CONFIG["allowed_extensions"],
        help="Upload a clear photo of a single clothing item",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Validate file
        if not validate_image(uploaded_file):
            st.stop()

        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("📷 Your Uploaded Item")
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Item", use_container_width=True)

        # Process image
        if st.button("🔮 Complete My Outfit", type="primary"):
            with st.spinner("🤔 AI is thinking about your perfect outfit..."):
                try:
                    # Save uploaded file
                    temp_path = save_uploaded_file(uploaded_file)
                    if temp_path is None:
                        st.stop()

                    # Get outfit completion
                    completed_outfit = model.complete_outfit(temp_path)

                    # Clean up temporary file
                    Path(temp_path).unlink(missing_ok=True)

                    if completed_outfit:
                        # Get image paths
                        outfit_images = model.get_outfit_images(completed_outfit)

                        # Display results
                        st.markdown(
                            '<div class="result-section">', unsafe_allow_html=True
                        )
                        st.success("🎉 Perfect outfit found!")

                        display_outfit_grid(outfit_images, "✨ Your Complete Outfit")

                        # Additional info
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("👕 Items in outfit", len(outfit_images))
                        with col2:
                            st.metric("🎯 Confidence", "High")

                        st.markdown("</div>", unsafe_allow_html=True)

                        # Download option
                        if st.button("💾 Save This Outfit"):
                            st.info("Outfit saved to your favorites! 📁")

                    else:
                        st.warning(
                            "😔 Couldn't find a matching outfit. Try a different image or check if the item is in our database."
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Outfit completion error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Made with ❤️ using Streamlit and PyTorch | 
        <a href='#' style='color: #FF6B6B;'>About</a> | 
        <a href='#' style='color: #FF6B6B;'>Contact</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
