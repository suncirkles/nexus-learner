"""Probe app: render a progress bar and button to discover AppTest attribute names."""
import streamlit as st
st.progress(0.5, text="half way")
st.button("click me")
