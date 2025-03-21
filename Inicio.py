import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Isótopos Estables")

logo = Image.open("images/PageImage.png")  # Reemplaza con la ruta de tu logo

# Mostrar el logo en la barra lateral
st.sidebar.image(logo, use_container_width=True)


with st.sidebar:
    st.markdown("""
        # Uso de isótopos en Hidrogeología
    """)
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Ajusta el número de <br> según necesites

    st.sidebar.image("images/Profile.png", width=80)
    # Texto en la barra lateral
    st.sidebar.markdown(
        """
        <div style="text-align: left;">
            <p><strong>Prof. Adriana Piña</strong><br> Universidad Nacional de Colombia</p>
        </div>
        """,
        unsafe_allow_html=True
    )