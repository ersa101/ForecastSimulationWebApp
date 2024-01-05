import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define session state outside of main function
if 'start_month' not in st.session_state:
    st.session_state.start_month = "01/2024"
if 'forecast_duration' not in st.session_state:
    st.session_state.forecast_duration = 36
if 'grouping_frequency' not in st.session_state:
    st.session_state.grouping_frequency = "Monthly"
if 'iternum' not in st.session_state:
    st.session_state.iternum = 0
    
def initial():
    st.title("Forecasting Input Dashboard : Initial Parameters")

    start_month = st.text_input("Enter start month for forecast (MM/YYYY):", "01/2024")
    forecast_duration = st.number_input("Enter the number of months for forecast:", min_value=1, value=36)
    grouping_frequency_options = ["Monthly", "Quarterly", "Bi-Annual", "Annual"]
    grouping_interval = st.selectbox("Select grouping frequency:", grouping_frequency_options, key='grouping_interval')
    iternum = st.number_input("Enter the attempt number:", min_value = 1)

    # Save inputs to session
    st.session_state.start_month = start_month
    st.session_state.forecast_duration = forecast_duration
    st.session_state.grouping_frequency = grouping_interval
    st.session_state.iternum = iternum

           
    # Display the sum of all final rates
    st.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)
    cols = st.columns([1,6,1])
    with cols[1]:
        if st.button("Input Escalation Parameters"):
            st.success("Escalating You ➡️➡️➡️ ")
            st.session_state.current_page = 'esc'
            st.rerun()               
    

# if __name__ == "__main__":
#     initial()
