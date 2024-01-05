import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from functions import predata, baselease_addons, forecast_logic, grouped_dwnld_excel, format_with_commas

def escalation():
    st.title("Forecasting Input Dashboard : Escalation Parameters")

    # Initialize session_state if not present
    if 'parameters_df' not in st.session_state:
        st.session_state.parameters_df = pd.DataFrame(columns=['Parameter', 'Start Month', 'Value', 'Escalation Frequency'])
    if 'parameters_df2' not in st.session_state:
        st.session_state.parameters_df2 = pd.DataFrame(columns=['Location', 'Change Months', 'Base Rate', 'Addon Rate'])

    # Get Conditional Contributing parameters input
    with st.sidebar.expander("Input for DataFrame Conditional Contribution"):
        parameter_name_options = ["CPI change", "Fuel Escalation", "Electrical Escalation", "FX"]
                                  
        parameter_name = st.sidebar.selectbox("Select parameter name:", parameter_name_options, key='parameter_name')
                                  
        parameter_start_month = st.sidebar.text_input("Enter start month for parameter (MM/YYYY):", key='parameter_start_month')
                                  
        parameter_value = st.sidebar.number_input("Enter parameter value:", key='parameter_value')
                                  
        parameter_frequency_options = ["Monthly", "Quarterly", "Bi-Annual", "Annual"]
        parameter_frequency = st.sidebar.selectbox("Select parameter frequency:", parameter_frequency_options, key='parameter_frequency')
        
        # Display a "Save Changes" button
        st.sidebar.markdown('<div style="margin-bottom:10px;"></div>', unsafe_allow_html=True)
        save_changes = st.sidebar.button("Save/Upload Conditional Parameter")

        # Append the parameter to the DataFrame when "Save Changes" is clicked
        if save_changes:
            if parameter_start_month != '':
                new_parameter = {'Parameter': parameter_name, 'Start Month': parameter_start_month, 
                                 'Value': parameter_value, 'Escalation Frequency': parameter_frequency }
                st.session_state.parameters_df = pd.concat([st.session_state.parameters_df, pd.DataFrame([new_parameter])], ignore_index=True)
                st.sidebar.success("Input parameter added Successfully ✅ ")
            else:
                st.sidebar.error("Input parameter is incomplete ❌ ")
                
        
        # File uploader for CSV
        st.sidebar.markdown('<div style="margin-bottom:10px;"></div>', unsafe_allow_html=True)
        uploaded_esc = st.sidebar.file_uploader("Upload Escalation CSV file", type=["csv"])
        if uploaded_esc is not None:
            uploaded_esc = pd.read_csv(uploaded_esc)
            print(uploaded_esc)
            print(uploaded_esc.shape)
            st.session_state.parameters_df = pd.concat([st.session_state.parameters_df, uploaded_esc], ignore_index=True)
    
    # Display the updated DataFrame
    st.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)
    st.write("Conditional Contributional / Escalation Parameters")
    st.write(st.session_state.parameters_df)
    
    
    # Display the sum of all final rates
    st.markdown('<div style="margin-bottom:40px;"></div>', unsafe_allow_html=True)
    cols = st.columns([1,3,1,3,1])
    with cols[1]:
        if st.button("Input Base Lease Parameters"):
            st.success("Taking you to Base Lease ➡️➡️➡️ ")
            st.session_state.current_page = 'baselease'
            st.rerun()               
    
    with cols[3]:
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
#     escalation()
