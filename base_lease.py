import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from functions import predata, baselease_addons, forecast_logic, grouped_dwnld_excel, format_with_commas

def baselease():
    st.title("Forecasting Input Dashboard : Base Lease Parameters")

    # Initialize session_state if not present
    if 'parameters_df2' not in st.session_state:
        st.session_state.parameters_df2 = pd.DataFrame(columns=['Location', 'Change Months', 'Base Rate', 'Addon Rate'])

    # Get Base Lease parameters input                                                   
    with st.sidebar.expander("Input for Base Lease Changes"):
        location_name_options = ['Colocation R1 Grid', 'Colocation R2 Grid', 'Colocation R3 Grid', 
                                'Colocation R1 Off Grid', 'Colocation R2 Off Grid', 'Colocation R3 Off Grid']
        location_name = st.sidebar.selectbox("Select Location:", location_name_options, key='location_name')
        
        parameter_change_month = st.sidebar.text_input("Enter start month for parameter (MM/YYYY):", key='parameter_change_month')
        
        parameter_baselease_value = st.sidebar.number_input("Enter Base Lease value:", key='parameter_baselease_value')
        
        parameter_addon_value = st.sidebar.number_input("Enter WindLoad unit value (in percent):", key='parameter_addon_value')
        
        # Display a "Save Changes" button
        st.sidebar.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)
        save_changes = st.sidebar.button("Save Base Lease Changes")

        # Append the parameter to the DataFrame when "Save Changes" is clicked
        if save_changes:
            if parameter_change_month != '':
                baselease_parameter = {'Location': location_name,'Change Months': parameter_change_month, 
                                     'Base Rate': parameter_baselease_value, 'Addon Rate': parameter_addon_value }
                st.session_state.parameters_df2 = pd.concat([st.session_state.parameters_df2, pd.DataFrame([baselease_parameter])], ignore_index=True)
                st.sidebar.success("Input parameter added Successfully ✅ ")
            else:
                st.sidebar.error("Input parameter is incomplete ❌ ")

        # File uploader for CSV
        st.sidebar.markdown('<div style="margin-bottom:10px;"></div>', unsafe_allow_html=True)
        uploaded_baselease = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_baselease is not None:
            print(uploaded_baselease.shape)
            st.session_state.parameters_df2 = pd.concat([st.session_state.parameters_df2, pd.Dataframe(uploaded_baselease)], ignore_index=True)
            
            
    # Display the updated DataFrame
    st.write("Base Lease & AddOn Rates")
    st.write(st.session_state.parameters_df2)
    
    # Add some margin before the "Forecast Now" button
    st.markdown('<div style="margin-bottom:20px;"></div>', unsafe_allow_html=True)
    
    # Display the sum of all final rates
    cols = st.columns([1, 6, 1])
    with cols[1]:
        if st.button("Forecast Now 🧮"):
            st.success("Moving you to Forecast ➡️➡️➡️ ")
            
            st.write(f"Performing forecast for {st.session_state.forecast_duration} months starting from {st.session_state.start_month}")

            # Run the Simulations
            visua, base, plot_col = forecast_logic(st.session_state.parameters_df, st.session_state.start_month, 
                                                   st.session_state.forecast_duration, st.session_state.parameters_df2)
            print("forecast R*C is ", visua.shape, "\n", visua.columns, "\n")
            visua_grp, visua_grp_perc = grouped_dwnld_excel(visua, st.session_state.grouping_frequency)
            print(visua_grp.shape, "and grouped is ", visua_grp_perc.shape, "\n")
            
            st.session_state.visua = visua
            st.session_state.base = base
            st.session_state.plot_col = plot_col
            st.session_state.visua_grp = visua_grp
            st.session_state.visua_grp_perc = visua_grp_perc
            
            with pd.ExcelWriter(f"visua_allData_iter{int(round(st.session_state.iternum, 0))}.xlsx") as writer:
                st.session_state.parameters_df.to_excel(writer, sheet_name="conditional params", index=False)
                st.session_state.parameters_df2.to_excel(writer, sheet_name="base lease params", index=False)
                st.session_state.base.to_excel(writer, sheet_name="base", index=False)
                st.session_state.visua.applymap(format_with_commas).to_excel(writer, sheet_name="visua", index=False)
                st.session_state.visua_grp.applymap(format_with_commas).to_excel(writer, sheet_name="visua_grouped", index=False)
                st.session_state.visua_grp_perc.to_excel(writer, sheet_name="visua_grouped_perc", index=False)


            st.session_state.current_page = 'plot'
            st.rerun()

# if __name__ == "__main__":
#     baselease()
