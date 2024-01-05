import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import first_page
import escalation_page
import base_lease
import plot_download

def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'first'

    if st.session_state.current_page == 'first':
        first_page.initial()
    elif st.session_state.current_page == 'esc':
        escalation_page.escalation()
    elif st.session_state.current_page == 'baselease':
        base_lease.baselease()
    elif st.session_state.current_page == 'plot':
        plot_download.plot()

if __name__ == "__main__":
    main()
