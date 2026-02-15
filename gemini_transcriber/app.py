# Import Streamlit for building the web UI
import streamlit as st

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import OS module to access environment variables
import os

# Import regex module for extracting YouTube video ID
import re

# Import YouTube transcript API to fetch subtitles
from youtube_transcript_api import YouTubeTranscriptApi

# Import specific transcript-related errors for better handling
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Import Gemini SDK (new unified SDK)
from google import genai


# ---------------------------------------------------------
# STEP 1: Load Environment Variables
# ---------------------------------------------------------

# Load variables from .env file into system environment
load_dotenv()

# Get Google API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY")

# If API key is missing, stop execution
if not API_KEY:
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()

# Create Gemini client using the API key
# This client will be reused for all model requests
client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------
# STEP 2: Define Prompt for Gemini
# ---------------------------------------------------------

# This prompt instructs Gemini how to summarize the transcript
PROMPT = """
You are a professional YouTube video summarizer.

Summarize the transcript below into clear bullet points.
Keep it under 250 words.
Focus only on the most important ideas.
"""


# ---------------------------------------------------------
# STEP 3: Extract Video ID from Any YouTube URL
# ---------------------------------------------------------

def get_video_id(url):
    """
    Extracts the 11-character YouTube video ID
    from different possible YouTube URL formats.
    """

    # Regex pattern to match YouTube video ID
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------
# STEP 4: Fetch Transcript from YouTube
# ---------------------------------------------------------

def extract_transcript_details(youtube_video_url):
    """
    Fetches transcript text from YouTube using video ID.
    Handles possible transcript-related errors.
    """

    try:
        # Extract video ID
        video_id = get_video_id(youtube_video_url)

        if not video_id:
            st.error("Invalid YouTube URL.")
            return None

        # Create YouTube Transcript API object
        api = YouTubeTranscriptApi()

        # Fetch transcript
        transcript = api.fetch(video_id)

        # Combine all transcript segments into one large string
        transcript_text = " ".join([item.text for item in transcript])

        return transcript_text

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None

    except NoTranscriptFound:
        st.error("No transcript available for this video.")
        return None

    except Exception as e:
        st.error(f"Transcript Error: {e}")
        return None


# ---------------------------------------------------------
# STEP 5: Generate Summary Using Gemini
# ---------------------------------------------------------

def generate_gemini_content(transcript_text):
    """
    Sends transcript text to Gemini model
    and returns summarized response.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Fast and cost-efficient model
            contents=PROMPT + transcript_text
        )

        return response.text

    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None


# ---------------------------------------------------------
# STEP 6: Streamlit UI Section
# ---------------------------------------------------------

# Set page title
st.title("🎥 YouTube Transcript to Detailed Notes Converter")

# Input field for user to enter YouTube link
youtube_link = st.text_input("Enter YouTube Video Link:")

# If user pastes a link, show video thumbnail
if youtube_link:
    video_id = get_video_id(youtube_link)

    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=600)
    else:
        st.warning("Could not extract video ID.")

# When user clicks button
if st.button("Get Detailed Notes"):

    if not youtube_link:
        st.warning("Please enter a YouTube link first.")

    else:
        # Show loading spinner while processing
        with st.spinner("Fetching transcript and generating summary..."):

            # Step 1: Fetch transcript
            transcript_text = extract_transcript_details(youtube_link)

            if transcript_text:

                # Step 2: Send transcript to Gemini
                summary = generate_gemini_content(transcript_text)

                if summary:
                    st.markdown("## 📝 Detailed Notes:")
                    st.write(summary)
