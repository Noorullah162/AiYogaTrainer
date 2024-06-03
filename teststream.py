import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # You can do any transformation on the frame here
        return frame

def main():
    st.title("Live Webcam Stream with OpenCV and Streamlit")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
