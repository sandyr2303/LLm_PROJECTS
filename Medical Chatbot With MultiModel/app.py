import streamlit as st
import os
from PIL import Image
import io
import logging
import base64
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot Multi-Model Educational purpose",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

def make_api_request(model, base64_image, query):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return {"error": "GROQ_API_KEY not found in environment variables"}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages
            },
            timeout=60 # Increased timeout
        )
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return None

def process_image(image_file, query):
    try:
        # Read image bytes
        image_data = image_file.getvalue()
        
        # Verify image
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            # Re-open for display since verify() moves cursor
            img = Image.open(io.BytesIO(image_data)) 
            img.verify()
        except Exception as e:
            return {"error": f"Invalid image: {str(e)}"}, img

        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Create two placeholders for the results
        col1, col2 = st.columns(2)
        
        models = [
            ("Llama-4 Scout (17B)", "meta-llama/llama-4-scout-17b-16e-instruct"),
            ("Llama-4 Maverick (90B)", "meta-llama/llama-4-maverick-17b-128e-instruct") # Note: User's code said llama-4-maverick, keeping it but commenting it's likely bigger
        ]

        results = {}

        with st.spinner('Analyzing with AI models...'):
            # Model 1
            response1 = make_api_request(models[0][1], encoded_image, query)
            if response1 and response1.status_code == 200:
                results[models[0][0]] = response1.json()["choices"][0]["message"]["content"]
            else:
                error_msg = response1.text if response1 else "Request failed"
                results[models[0][0]] = f"Error: {error_msg}"

            # Model 2
            response2 = make_api_request(models[1][1], encoded_image, query)
            if response2 and response2.status_code == 200:
                results[models[1][0]] = response2.json()["choices"][0]["message"]["content"]
            else:
                error_msg = response2.text if response2 else "Request failed"
                results[models[1][0]] = f"Error: {error_msg}"
                
        return results, img

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}, None

def main():
    st.markdown('<h1 class="main-header">🏥 Medical Image Analysis Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Upload a medical image and ask questions to get insights from multiple AI models simultaneously.")

    # Sidebar for controls
    with st.sidebar:
        st.header("Configuration")
        st.info("Ensure your GROQ_API_KEY is set in the .env file.")
        
        uploaded_file = st.file_uploader("Upload Medical Image", type=['png', 'jpg', 'jpeg'])
        
    # Main content area
    if uploaded_file:
        # Layout: Image on top (centered or taking reasonable space), then Query, then Results
        
        # Display Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        # Query Input
        query = st.text_area("Ask a question about the image:", value="Analyze this medical image and identify any anomalies.", height=100)
        
        analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)
        
        if analyze_button:
            if not query:
                st.warning("Please enter a query.")
            else:
                # Debug Info
                # st.write("Debug: Starting analysis...")
                api_key_exists = bool(os.getenv('GROQ_API_KEY'))
                # st.write(f"Debug: GROQ_API_KEY present: {api_key_exists}")
                
                results, _ = process_image(uploaded_file, query)
                
                # st.write("Debug: Processing complete. Results:", results)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    # Output for Model 1
                    with col1:
                        st.markdown('<div class="model-header">Llama-4 Scout</div>', unsafe_allow_html=True)
                        response_text = results["Llama-4 Scout (17B)"]
        
                        st.markdown(f'<div class="result-box" style="background-color: black">{response_text}</div>', unsafe_allow_html=True)
                        if "Error" in response_text:
                            st.error(response_text)
                        
                    # Output for Model 2
                    with col2:
                        st.markdown('<div class="model-header">Llama-4 Maverick</div>', unsafe_allow_html=True)
                        response_text = results["Llama-4 Maverick (90B)"]
    
                        st.markdown(f'<div class="result-box" style="background-color: black">{response_text}</div>', unsafe_allow_html=True)
                        if "Error" in response_text:
                            st.error(response_text)

    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
