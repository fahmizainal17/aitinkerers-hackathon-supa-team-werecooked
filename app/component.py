import streamlit as st
from PIL import Image
import base64

def get_base64_of_bin_file(bin_file):
    """
    Function to encode local file (image or gif) to base64 string
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def page_style():
    # Encode the local GIF to base64

    sidebar_background_base64 = get_base64_of_bin_file('assets/Judge_Background.jpg')

    # Apply custom styles, including the sidebar background GIF
    custom_style = f"""
        <style>
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}

            /* Sidebar background with a dark overlay */
            [data-testid="stSidebar"] > div:first-child {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)), 
                                  url("data:image/jpeg;base64,{sidebar_background_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: local;
            }}

            [data-testid="stHeader"] {{
                background: rgba(0,0,0,0);
            }}

            [data-testid="stToolbar"] {{
                right: 2rem;
            }}

            .stButton>button {{
                background-color: #4CAF50; 
                color: white !important;
            }}

            .stDownloadButton>button {{
                background-color: #4CAF50; 
                color: white !important;
            }}

            /* Certification Card Styles */
            .cert-card {{
                background-color: #333333;
                color: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}

            .cert-card:hover {{
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }}
        </style>
    """

    # Set the page configuration
    icon = Image.open('photos/werecooked_round_profile.png')
    st.set_page_config(page_title="Fahmi Zainal", page_icon=icon, layout="wide")

    # Apply custom styles to the page
    st.markdown(custom_style, unsafe_allow_html=True)

    # Display the main background image
    image = Image.open('app/photos/werecooked_background.png')
    st.image(image)

    # Sidebar content
    with st.sidebar:
        # Display the round profile picture at the top of the sidebar
        st.image("photos/werecooked_round_profile.png", width=150)

        st.markdown("""
            ## Created By: Fahmi Zainal
            
        """)

        st.markdown("""
            ## Skills You Will Learn 
            - **Analysis and Visualization**
            - **Streamlit Dashboarding**
            - **Python (Programming Language)**
            - **Streamlit Web Development**
        """)

        # HTML and JavaScript to open YouTube in a new tab
        new_tab_button = """
        <a href="https://www.youtube.com/watch?v=VeUiVCb7ZmQ?si=GzSBUP3zs1hEkigI" target="_blank">
            <button style="background-color: #4CAF50; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                🎵 Play Music while Reading
            </button>
        </a>
        """
        st.markdown(new_tab_button, unsafe_allow_html=True)

        # LinkedIn button with logo
        linkedin_url = "https://www.linkedin.com/in/fahmizainal17"
        st.markdown(f"""
            <a href="{linkedin_url}" target="_blank">
                <button style="background-color: #0077B5; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/1200px-LinkedIn_icon.svg.png" width="16" style="vertical-align: middle;"> Connect on LinkedIn
                </button>
            </a>
        """, unsafe_allow_html=True)

        # GitHub button with logo
        github_url = "https://github.com/fahmizainal17"
        st.markdown(f"""
            <a href="{github_url}" target="_blank">
                <button style="background-color: #333; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="16" style="vertical-align: middle;"> Check out my GitHub
                </button>
            </a>
        """, unsafe_allow_html=True)