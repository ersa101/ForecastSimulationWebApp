import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from functions import format_with_commas

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot():
    st.title("Forecasting Results")

    # Plots
    plot_names = ['Grid Locations Revenue', 'OffGrid Locations Revenue', 'Total Revenue', 
                  'Escalations', 'Base Rate Variations', 'AddOn Variations']

    fig = make_subplots(rows=3, cols=2, subplot_titles=[f"Plot for {i}" for i in plot_names])

    for i, set_columns in enumerate(st.session_state.plot_col):
        row = i // 2 + 1
        col = i % 2 + 1
        for col_name in set_columns:
            fig.add_trace(go.Scatter(x = st.session_state.visua['months'], y = st.session_state.visua[col_name], 
                                     mode='lines+markers', name=col_name),
                          row=row, col=col)
            fig.update_xaxes(tickangle=-90, showgrid=True, gridcolor='lightgray', row=row, col=col)
            fig.update_yaxes(showgrid=False, row=row, col=col)

    fig.update_layout(height=1200, showlegend=True, plot_bgcolor='white',
                      legend=dict(orientation='h', x=0, y=-0.25, traceorder='normal'),
                      margin=dict(l=10, r=10, t=70, b=70))

    # Buttons in the left sidebar
    show_graph = st.sidebar.button("Show Graph 📈")
    show_table = st.sidebar.button("Show Table 📅")
    download_button = st.sidebar.button("Download XLSX 📑")
    
    st.sidebar.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)
    restart = st.sidebar.button("Start Again 🤯")

    # Show Graph button
    if show_graph:
        st.plotly_chart(fig)

    # Show Table button
    if show_table:
        st.dataframe(st.session_state.visua_grp)

    # Download XLSX button
    if download_button:
        st.success("Downloading your Compiled Data File ✌️✌️✌️")
        with pd.ExcelWriter(f"visua_allData_iter{int(round(st.session_state.iternum, 0))}.xlsx") as writer:
            st.session_state.parameters_df.to_excel(writer, sheet_name="conditional params", index=False)
            st.session_state.parameters_df2.to_excel(writer, sheet_name="base lease params", index=False)
            st.session_state.base.to_excel(writer, sheet_name="base", index=False)
            st.session_state.visua.applymap(format_with_commas).to_excel(writer, sheet_name="visua", index=False)
            st.session_state.visua_grp.applymap(format_with_commas).to_excel(writer, sheet_name="visua_grouped", index=False)
            st.session_state.visua_grp_perc.to_excel(writer, sheet_name="visua_grouped_perc", index=False)
            
    # Show Table button
    if restart:
        st.success("Taking you back to Initialization Screen 🔁🔁🔁")
        st.session_state.current_page = 'first'
        st.rerun()

# if __name__ == "__main__":
#     plot()
