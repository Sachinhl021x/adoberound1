"""Custom CSS styles for Adobe branding."""

ADOBE_CSS = """
<style>
    /* Adobe Fonts (using system sans-serif fallback closely matching Adobe Clean) */
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }

    /* Primary Color (Adobe Red) overrides */
    :root {
        --primary-color: #FA0F00;
        --background-color: #FFFFFF;
        --secondary-background-color: #F5F5F5;
        --text-color: #2C2C2C;
        --font: "Segoe UI", sans-serif;
    }

    /* Streamlit Main Menu & Footer Hide */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Header */
    .adobe-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 2px solid #FA0F00;
        margin-bottom: 2rem;
    }
    
    .adobe-logo-text {
        font-weight: 800;
        font-size: 24px;
        color: #FA0F00;
        margin-left: 10px;
        letter-spacing: -0.5px;
    }

    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent !important;
    }

    /* User Message Bubble */
    div[data-testid="chat-message-user"] {
        background-color: #F0F0F0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Assistant Message Bubble */
    div[data-testid="chat-message-assistant"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FA0F00 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #D40C00 !important;
        box-shadow: 0 4px 8px rgba(250, 15, 0, 0.2);
    }
    
    /* Citations / Expanders */
    .streamlit-expanderHeader {
        background-color: #F8F9FA !important;
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
        font-weight: 500 !important;
    }
    
    /* Links */
    a {
        color: #FA0F00 !important;
        text-decoration: none !important;
    }
    a:hover {
        text-decoration: underline !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #999;
    }
</style>
"""

def apply_custom_styles():
    """Apply the custom CSS to the Streamlit app."""
    import streamlit as st
    st.markdown(ADOBE_CSS, unsafe_allow_html=True)

def render_header():
    """Render the custom Adobe-themed header."""
    import streamlit as st
    st.markdown("""
        <div class="adobe-header">
            <svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px"
                 viewBox="0 0 100 100" style="enable-background:new 0 0 100 100; width: 40px; height: 40px;" xml:space="preserve">
            <style type="text/css">
                .st0{fill:#FA0F00;}
                .st1{fill:#FFFFFF;}
            </style>
            <rect class="st0" width="100" height="100"/>
            <polygon class="st1" points="68.8,17.4 86.8,17.4 86.8,82.6 68.8,37.3 "/>
            <polygon class="st1" points="13.2,17.4 31.2,17.4 13.2,82.6 "/>
            <polygon class="st1" points="48.5,17.4 51.5,17.4 62.1,82.6 54.8,82.6 40.6,35.2 36.1,51.8 52.9,51.8 	"/>
            </svg>
            <span class="adobe-logo-text">Leadership Insight Agent</span>
        </div>
    """, unsafe_allow_html=True)
