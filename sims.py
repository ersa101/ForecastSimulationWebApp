import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def predata(conds, start_date, forecasting_period):
    
    ## Generate monthly date values for forecasting period
    start_date = pd.to_datetime(start_date, format='%m/%Y')
    end_date = start_date + relativedelta(months = forecasting_period-1)
    date_values = []
    current_date = start_date
    while current_date <= end_date:
        date_values.append(current_date)
        current_date = current_date + relativedelta(months = 1)
    months = pd.DataFrame(date_values, columns=['months'])
    print(months.shape)
    print(months, "\n")

    ## Frequency options
    freq = pd.DataFrame({'Escalation Frequency':['Monthly', 'Quarterly' ,'Bi-Annual', 'Annual'], 'freq_number':[1,3,6,12]})
    print(freq.shape)
    print(freq, "\n")

    ## usable parameter / lists
    params = ["FX", "CPI change", "Fuel Escalation", "Electrical Escalation", "Lease Rate"]
    contri_params = ["CPI change", "Fuel Escalation", "Electrical Escalation"]
    cond_params = conds.Parameter.unique()
    imp_uncond_params = set(contri_params) - set(cond_params)


    # creating a hard coded previous values table     #####  NEED TO AUTOMATE IT
    ## contributing params
    prev_vals =  pd.DataFrame(columns=['Parameter', 'Start Month', 'percent', 'Escalation Frequency', 'freq_number', 'months', 'GridVal', 'OffGridVal'])
    for l in params:
        prev_vals = prev_vals.append({'Parameter': l, 
                                      'months': (months.months.min() - relativedelta(months = 1)).strftime('%Y-%m-%d')}
                                     , ignore_index=True)

    prev_vals.loc[(prev_vals.Parameter == 'CPI change'), 'GridVal'] = 0.753
    prev_vals.loc[(prev_vals.Parameter == 'CPI change'), 'OffGridVal'] = 0.645
    prev_vals.loc[(prev_vals.Parameter == 'CPI change'), 'percent'] = 6.45

    prev_vals.loc[(prev_vals.Parameter == 'Fuel Escalation'), 'GridVal'] = 0.113
    prev_vals.loc[(prev_vals.Parameter == 'Fuel Escalation'), 'OffGridVal'] = 0.451
    prev_vals.loc[(prev_vals.Parameter == 'Fuel Escalation'), 'percent'] = -4.20

    prev_vals.loc[(prev_vals.Parameter == 'Electrical Escalation'), 'GridVal'] = 0.2
    prev_vals.loc[(prev_vals.Parameter == 'Electrical Escalation'), 'OffGridVal'] = 0
    prev_vals.loc[(prev_vals.Parameter == 'Electrical Escalation'), 'percent'] = 0
    
    return months, freq, prev_vals, params, contri_params, cond_params, imp_uncond_params




def baselease_addons(base_lease, conds, start_date, forecasting_period):
        
    months, freq, prev_vals, params, contri_params, cond_params, imp_uncond_params = predata(conds, start_date, forecasting_period)

    ## Previous base lease rate table for all locations
    prev_loc_rates = pd.DataFrame({'loca': ['Colocation R1 Grid', 'Colocation R2 Grid', 'Colocation R3 Grid', 
                                    'Colocation R1 Off Grid', 'Colocation R2 Off Grid', 'Colocation R3 Off Grid'],
                            'months': pd.to_datetime('2023-12-01', format='%Y-%m-%d'),
                            'Base Rate': [4095, 4095, 4095, 4095, 4095, 4095],
                            'Addon Rate': [10, 15, 20, 25, 30, 35]})
    print(prev_loc_rates.shape, "\n", prev_loc_rates, "\n")

    locs = prev_loc_rates.loca.unique()
    changed_locs = base_lease.Location.unique()
    unchanged_locs = set(locs) - set(changed_locs)
    print("locations data :", locs, "\n", changed_locs, "\n", unchanged_locs, "\n")
    loc_tab_cols = ['Location', 'months', 'Base Rate', 'Addon Rate']
    loc_tab = pd.DataFrame(columns = loc_tab_cols)
    base_lease['Change Months'] = pd.to_datetime(base_lease['Change Months'])
    base_lease = base_lease.sort_values(by=['Location', 'Change Months'], ascending = [True, True])

    ## those params which are important and contributing and changed in conditions provided
    for k in locs:
        if k in changed_locs:
            df = base_lease[base_lease.Location.isin([k])].merge(months, how = 'right', left_on = ['Change Months'], right_on = 'months')
            df = df.sort_values(by='months')
            print(k, df.shape, df.head(3))
            
            temp = prev_loc_rates[prev_loc_rates.loca == k]
            df2 = temp.append(df, ignore_index=True)
            df2['Base Rate'].fillna(0, inplace=True)
            df2['Addon Rate'].fillna(0, inplace=True)
            df2['Location'] = k
            for x in range(1, df3.shape[0]):
                if df3.iloc[x,2] == 0:
                    df3.iloc[x, 2] = df3.iloc[x-1,2]
                    df3.iloc[x, 3] = df3.iloc[x-1,3]  
            print("df3", df3.head(10), "\n") 

            loc_tab = loc_tab.append(df3, ignore_index=True)
            print("loc_tab", loc_tab.shape, "\n")
            
    for p in list(unchanged_locs):
        df2 = prev_loc_rates[prev_loc_rates.loca.isin([p])].merge(months, how = 'right', on = 'months')
        df2 = df2.sort_values(by='months')
        print(p, df2.shape, df2.head(3))

        df2['Base Rate'].fillna(0, inplace=True)
        df2['Addon Rate'].fillna(0, inplace=True)
        df2['Location'] = p

        for x in range(1, df2.shape[0]):
            if df2.iloc[x,2] == 0:
                df2.iloc[x, 2] = df2.iloc[x-1,2]
                df2.iloc[x, 3] = df2.iloc[x-1,3]  
        print("df2", df2.head(10), "\n") 

        loc_tab = loc_tab.append(df2, ignore_index=True)
        print("loc_tab", loc_tab.shape, "\n")
            

    loc_tab['rate'] = loc_tab['Base Rate'] * loc_tab['Addon Rate']
    loc_tab['months'] = pd.to_datetime(loc_tab['months'])
    loc_tab['months'] = loc_tab['months'].dt.strftime('%Y-%m-%d')
    print("final loc_tab", loc_tab.shape, "\n Which is ", loc_tab.shape[0] == ((months.shape[0] + 1) * len(locs)))
    return loc_tab


    
def forecast_logic(conds, start_date, forecasting_period, base_lease):

    months, freq, prev_vals, params, contri_params, cond_params, imp_uncond_params  = predata(conds, start_date, forecasting_period)
    loc_tab = baselease_addons(base_lease, conds, start_date, forecasting_period)
    
    base = months.copy()
    
    ## conditional table
    print(conds.shape)
    conds['Start Month'] = pd.to_datetime(conds['Start Month'])
    conds = conds.sort_values(by=['Parameter', 'Start Month'], ascending = [True, True])
    print(conds, "\n")
    
    contri_GridVal, contri_OffGridVal, cummpercs = [], [], ['months']
    
    # generating a table with imp_conds monthly variation
    ## those params which are important and contributing and changed in conditions provided
    for k in params:
        if k in cond_params:
            df = conds[conds.Parameter.isin([k])].merge(freq, how = 'left', on='Escalation Frequency')
            df2 = df.merge(months, how = 'right', left_on = ['Start Month'], right_on = 'months')
            df2.rename(columns={'Value':'percent'}, inplace=True)
            df2 = df2.sort_values(by='months')
            print("\n", k, "\n", df2)

            ##############################################################################################################

            ## creating a filtered table for various month change
            chk = df.copy()
            chk = chk.append({'Start Month':df2.months.max(), 'percent':0, 'freq_number':0}, ignore_index=True)
            chk['mnths'] = ''
            print("\n", k, "\n", chk)

            for i in range(chk.shape[0]-2):
                strt_dt, end_dt, interval = chk.iloc[i,1], chk.iloc[i+1,1], int(chk.iloc[i,4])
                date_values = []
                while strt_dt < end_dt:
                    strt_dt = strt_dt + relativedelta(months=interval)
                    date_values.append(strt_dt.strftime('%Y-%m-%d'))
                date_values = str(date_values[:-1])
                chk.iloc[i,6] = date_values

            for j in range(chk.shape[0]-2, chk.shape[0]-1):
                strt_dt, end_dt, interval = chk.iloc[j,1], chk.iloc[j+1,1], int(chk.iloc[j,4])
                print(chk.iloc[j,1], chk.iloc[j+1,1], chk.iloc[j,4])
                date_values = []
                while strt_dt <= end_dt:
                    strt_dt = strt_dt + relativedelta(months=interval)
                    date_values.append(strt_dt.strftime('%Y-%m-%d'))
                date_values = str(date_values[:-1])
                chk.iloc[j,6] = date_values

            chk['mnths'] = chk['mnths'].str.replace('[', '').str.replace(']', '').str.replace("'", '')

            ##############################################################################################################

            ## expanding offset column values
            dt_list =  chk[['mnths']].dropna()
            dt_list['num_elements'] = dt_list['mnths'].str.count(',') + 1

            expanded_df = pd.DataFrame()

            for index, row in dt_list.iterrows():
                dates_list = row['mnths'].split(', ')
                num_elements = row['num_elements']

                for i in range(num_elements):
                    expanded_df = expanded_df.append({'mnths': dates_list[i], 'original_mnths': row['mnths']}, ignore_index=True)

            dt_list = dt_list.drop(['mnths', 'num_elements'], axis=1)
            result_df = pd.concat([dt_list, expanded_df], ignore_index=True).dropna()
            result_df.rename(columns={'mnths':'nxt_mnths', 'original_mnths':'mnths'}, inplace=True)

            ##############################################################################################################

            ## mapping expanded dates with the base file
            chk2 = chk.merge(result_df, how = 'left', on = 'mnths')
            chk2['nxt_mnths'] = pd.to_datetime(chk2['nxt_mnths'])

            chk3 = df2.merge(chk2, how='left', left_on='months', right_on='nxt_mnths')
            
            chk3['percent'] = np.where((chk3.percent_x.isna()) & (chk3.Value.isna()), np.NaN,
                                       np.where((chk3.percent_x.isna()),chk3.Value, chk3.percent_x))
            chk3['Parameter'] = np.where((chk3.Parameter_x.isna()) & (chk3.Parameter_y.isna()), np.NaN,
                                       np.where((chk3.Parameter_x.isna()),chk3.Parameter_y, chk3.Parameter_x))
            chk3['freq_number'] = np.where((chk3.freq_number_x.isna()) & (chk3.freq_number_y.isna()), np.NaN,
                                       np.where((chk3.freq_number_x.isna()),chk3.freq_number_y, chk3.freq_number_x))
            chk3['Escalation Frequency'] = np.where((chk3['Escalation Frequency_x'].isna()) & (chk3['Escalation Frequency_y'].isna()), np.nan,
                                       np.where((chk3['Escalation Frequency_x'].isna()),chk3['Escalation Frequency_y'], chk3['Escalation Frequency_x']))
            chk3['Start Month'] = np.where((chk3['Start Month_x'].isna()) & (chk3['Start Month_y'].isna()), np.datetime64("NaT"),
                                       np.where((chk3['Start Month_x'].isna()),chk3['Start Month_y'], chk3['Start Month_x']))

            chk4 = chk3[['Parameter', 'Start Month', 'percent', 'Escalation Frequency', 'freq_number', 'months']]
            chk4['GridVal'] = np.NaN
            chk4['OffGridVal'] = np.NaN

            temp = prev_vals[prev_vals.Parameter==k]
            chk5 = temp.append(chk4, ignore_index=True)
            chk5.percent.fillna(0, inplace=True)
            chk5['CummPerc'] = chk5.percent.cumsum()
            chk5['FinalPerc'] = np.where(pd.isna(chk5.percent), np.NaN, chk5.percent)
            chk5['months'] = pd.to_datetime(chk5['months'])
            chk5['months'] = chk5['months'].dt.strftime('%Y-%m-%d')

            for i in range(1, chk5.shape[0]):
                if chk5.iloc[i, chk5.columns.get_loc('percent')] != 0:
                    chk5.iloc[i, chk5.columns.get_loc('FinalPerc')] = chk5.iloc[i, chk5.columns.get_loc('percent')]
                else:
                    chk5.iloc[i,chk5.columns.get_loc('FinalPerc')] = float(chk5.iloc[i-1, chk5.columns.get_loc('FinalPerc')])
                    
                    
                if pd.isna(chk5.iloc[i,chk5.columns.get_loc('freq_number')]):
                    chk5.iloc[i, chk5.columns.get_loc('GridVal')] = chk5.iloc[i-1, chk5.columns.get_loc('GridVal')]
                    chk5.iloc[i, chk5.columns.get_loc('OffGridVal')] = chk5.iloc[i-1, chk5.columns.get_loc('OffGridVal')]
                else:
                    chk5.iloc[i, chk5.columns.get_loc('GridVal')] = float(chk5.iloc[i-1, chk5.columns.get_loc('GridVal')]) * (1 + chk5.iloc[i, chk5.columns.get_loc('FinalPerc')] / 100)
                    chk5.iloc[i, chk5.columns.get_loc('OffGridVal')] = float(chk5.iloc[i-1, chk5.columns.get_loc('OffGridVal')]) * (1 + chk5.iloc[i, chk5.columns.get_loc('FinalPerc')] / 100)

                
            chk5.rename(columns={'GridVal':f"{k}_GridVal", 'OffGridVal':f"{k}_OffGridVal", 'CummPerc':f"{k}_CummPerc",
                                 'percent':f"{k}_percent", 'Escalation Frequency':f"{k}_freq", 'FinalPerc':f"{k}_FinalPerc",}
                        , inplace = True)
            
            contri_GridVal.append(f"{k}_GridVal")
            contri_OffGridVal.append(f"{k}_OffGridVal")
            cummpercs.append(f"{k}_CummPerc")

            chk6 = chk5[['months', f"{k}_percent", f"{k}_freq", f"{k}_GridVal", f"{k}_OffGridVal", f"{k}_CummPerc", f"{k}_FinalPerc"]]            
            chk6['months'] = pd.to_datetime(chk6['months'])
            base = base.merge(chk6, on='months', how = 'left')
            
            print("input conditions' name, shape & base forecast file shape resp. are :", k, chk6.shape, base.shape, "\n")


    ## those params which are important and contributing but not changed in conditions provided
    for h in list(imp_uncond_params):   
        temp = prev_vals[prev_vals.Parameter == h]
        temp_uncond_params =  months.copy()

        temp_uncond_params[f"{h}_percent"] = 0
        temp_uncond_params[f"{h}_CummPerc"] = prev_vals.loc[(prev_vals.Parameter == h), 'percent']
        temp_uncond_params[f"{h}_FinalPerc"] = 0
        temp_uncond_params[f"{h}_freq"] = 'Never'
        temp_uncond_params[f"{h}_GridVal"] = temp.GridVal.min()
        temp_uncond_params[f"{h}_OffGridVal"] = temp.OffGridVal.min()

        contri_GridVal.append(f"{h}_GridVal")
        contri_OffGridVal.append(f"{h}_OffGridVal")
        cummpercs.append(f"{h}_CummPerc")

        temp_uncond_params['months'] = pd.to_datetime(temp_uncond_params['months'])
        base = base.merge(temp_uncond_params, on='months', how = 'left')
        print("input conditions' name, shape & base forecast file shape resp. are :", h, temp_uncond_params.shape, base.shape, "\n")

    print("final forecast base table has R*C as ", base.shape)  

    
    ## create columns for sum of all major contributing factors
    base['sum_GridVal'] = base[contri_GridVal].sum(axis=1)
    base['sum_OffGridVal'] = base[contri_OffGridVal].sum(axis=1)
    print("final base table has R*C as ", base.shape)
    print("final base table has colnames as ", base.columns)
    
    ## joining the monthly base rates, addons, final rates with base
    pvt_loc_tab = loc_tab.pivot(index='months', columns='Location', values=['Base Rate', 'Addon Rate'])
    pvt_loc_tab = pvt_loc_tab.reset_index()
    print("colnames BEFORE pivot ", pvt_loc_tab.columns)
    pvt_loc_tab.columns = ['_'.join(col) for col in pvt_loc_tab.columns]
    pvt_loc_tab.rename(columns = {'months_':'months'}, inplace = True)
    print("colnames AFTER pivot ", pvt_loc_tab.columns)
    baserate_cols = [col for col in pvt_loc_tab.columns if 'Base' in col]
    addon_cols = [col for col in pvt_loc_tab.columns if 'Addon' in col]
    print("colnames having base rate values ", baserate_cols)
    print("colnames having addon values ", addon_cols)
    pvt_loc_tab['months'] = pd.to_datetime(pvt_loc_tab['months'])

    
    ## revenue table for all locations
    print("contri_GridVal cols :", contri_GridVal, "\n", "contri_OffGridVal cols :", contri_OffGridVal, "\n")
    print("loc_tab :", loc_tab.Location.value_counts(), "\n")
    rev_base = base[['months', 'sum_GridVal', 'sum_OffGridVal']]
    rev_tab = rev_base.merge(pvt_loc_tab, how = 'inner', on = 'months')
    rev_tab['Colocation R1 Grid'] = rev_tab.apply(lambda row: row['sum_GridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R1 Grid', 'rate'].values[0], axis=1)
    rev_tab['Colocation R2 Grid'] = rev_tab.apply(lambda row: row['sum_GridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R2 Grid', 'rate'].values[0], axis=1)
    rev_tab['Colocation R3 Grid'] = rev_tab.apply(lambda row: row['sum_GridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R3 Grid', 'rate'].values[0], axis=1)
    rev_tab['Colocation R1 Off Grid'] = rev_tab.apply(lambda row: row['sum_OffGridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R1 Off Grid', 'rate'].values[0], axis=1)
    rev_tab['Colocation R2 Off Grid'] = rev_tab.apply(lambda row: row['sum_OffGridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R2 Off Grid', 'rate'].values[0], axis=1)
    rev_tab['Colocation R3 Off Grid'] = rev_tab.apply(lambda row: row['sum_OffGridVal'] * loc_tab.loc[loc_tab['Location'] == 'Colocation R3 Off Grid', 'rate'].values[0], axis=1)

    gridlocs = ['Colocation R1 Grid', 'Colocation R2 Grid', 'Colocation R3 Grid']
    offgridlocs = ['Colocation R1 Off Grid', 'Colocation R2 Off Grid', 'Colocation R3 Off Grid']
    rev_tab['Grid Revenue'] = rev_tab[gridlocs].sum(axis=1)
    rev_tab['OffGrid Revenue'] = rev_tab[offgridlocs].sum(axis=1)
    rev_tab['Total Revenue'] = rev_tab['Grid Revenue'] + rev_tab['OffGrid Revenue']
    print("final REVENUE table size is ", rev_tab.shape, "\n")
    
    
    ## cummulative percentages for all params
    print("cummpercs cols :", cummpercs)
    cumm_perc = base[cummpercs]
    print(cumm_perc.shape)
    print(cumm_perc.columns)
    
    ## list of all plotting columns
    plot_col = [gridlocs, offgridlocs, {'Grid Revenue', 'OffGrid Revenue', 'Total Revenue'}, cummpercs[1:], baserate_cols, addon_cols]
    
    visua = rev_tab.merge(cumm_perc, on = 'months', how = 'inner')
    print("forecast R*C is ", rev_tab.shape, "\n", rev_tab.columns, "\n")
    print("cummulative escalations R*C is ", cumm_perc.shape, "\n", cumm_perc.columns, "\n")
    print(2*visua.shape[0] == (rev_tab.shape[0] + cumm_perc.shape[0]))
    
    visua['months'] = pd.to_datetime(visua['months'])
    visua['months'] = visua['months'].dt.strftime('%B-%y')
    visua.fillna(0, inplace=True)
    visua.iloc[:, 3:] = visua.iloc[:, 3:].astype(int)
    
    return visua, base, plot_col


def grouped_dwnld_excel(visua, grouping_frequency):
    ## grouping the rev sheet for display as per user selection
    rev = visua[['months', 'Colocation R1 Grid', 'Colocation R2 Grid', 'Colocation R3 Grid', 
                 'Colocation R1 Off Grid', 'Colocation R2 Off Grid', 'Colocation R3 Off Grid', 
                 'Grid Revenue', 'OffGrid Revenue', 'Total Revenue']]
    visua_grp = rev.copy()
    visua_grp['months'] = pd.to_datetime(visua_grp['months'], format = '%B-%y')
    visua_grp['monthnum'] = visua_grp['months'].dt.strftime('%m').astype(int)
    visua_grp['qtr_num'] = (visua_grp['monthnum'] - 0.5) // 3
    visua_grp['biannual_num'] = (visua_grp['monthnum'] - 0.5) // 6
    visua_grp['yearnum'] = visua_grp['months'].dt.strftime('%y').astype(int)

    if grouping_frequency == 'Quarterly':
        visua_grp = visua_grp.groupby(by = ['yearnum', 'qtr_num']).agg({'months':'min',
                                                     'Colocation R1 Grid':'sum', 'Colocation R2 Grid':'sum',
                                                     'Colocation R3 Grid':'sum', 'Colocation R1 Off Grid':'sum', 
                                                     'Colocation R2 Off Grid':'sum', 
                                                     'Colocation R3 Off Grid':'sum',
                                                     'Grid Revenue':'sum', 'OffGrid Revenue':'sum', 
                                                     'Total Revenue':'sum'}).reset_index()
        visua_grp.drop(['yearnum', 'qtr_num'], axis=1, inplace = True)
        visua_grp['months'] = pd.to_datetime(visua_grp['months']).dt.strftime('%B-%y')
    elif grouping_frequency == 'Bi-Annual':
        visua_grp = visua_grp.groupby(by = ['yearnum', 'biannual_num']).agg({'months':'min',
                                                     'Colocation R1 Grid':'sum', 'Colocation R2 Grid':'sum',
                                                     'Colocation R3 Grid':'sum', 'Colocation R1 Off Grid':'sum', 
                                                     'Colocation R2 Off Grid':'sum', 
                                                     'Colocation R3 Off Grid':'sum',
                                                     'Grid Revenue':'sum', 'OffGrid Revenue':'sum', 
                                                     'Total Revenue':'sum'}).reset_index()
        visua_grp.drop(['yearnum', 'biannual_num'], axis=1, inplace = True)
        visua_grp['months'] = pd.to_datetime(visua_grp['months']).dt.strftime('%B-%y')
    elif grouping_frequency == 'Annual':
        visua_grp = visua_grp.groupby(by = ['yearnum']).agg({'months':'min',
                                                     'Colocation R1 Grid':'sum', 'Colocation R2 Grid':'sum',
                                                     'Colocation R3 Grid':'sum', 'Colocation R1 Off Grid':'sum', 
                                                     'Colocation R2 Off Grid':'sum', 
                                                     'Colocation R3 Off Grid':'sum',
                                                     'Grid Revenue':'sum', 'OffGrid Revenue':'sum', 
                                                     'Total Revenue':'sum'}).reset_index()
        visua_grp.drop(['yearnum'], axis=1, inplace = True)
        visua_grp['months'] = pd.to_datetime(visua_grp['months']).dt.strftime('%B-%y')
    else:
        visua_grp = rev.copy()
    
    visua_grp_perc = visua_grp.copy()
    for a in range(1, visua_grp.shape[0]):
        for b in range(1,  visua_grp.shape[1]):
            visua_grp_perc.iloc[a,b] = str(round((visua_grp.iloc[a,b] - visua_grp.iloc[a-1, b]) * 100 / visua_grp.iloc[a-1, b], 2)) + " %"
            
    return visua_grp, visua_grp_perc




def main():
    st.title("Forecasting Input Dashboard")

    start_month = st.sidebar.text_input("Enter start month for forecast (MM/YYYY):", "01/2024")
    forecast_duration = st.sidebar.number_input("Enter the number of months for forecast:", min_value=1, value=36)
    grouping_frequency_options = ["Monthly", "Quarterly", "Bi-Annual", "Annual"]
    grouping_frequency = st.sidebar.selectbox("Select grouping frequency:", grouping_frequency_options, key='grouping_frequency')
    iternum = st.sidebar.number_input("Enter the attempt number:")

    input_sidebar = st.sidebar.container()
    st.sidebar.header("Parameters Input")

    # Initialize session_state if not present
    if 'parameters_df' not in st.session_state:
        st.session_state.parameters_df = pd.DataFrame(columns=['Parameter', 'Start Month', 'Value', 'Escalation Frequency'])
        st.session_state.parameters_df2 = pd.DataFrame(columns=['Location', 'Change Months', 'Base Rate', 'Addon Rate'])

    # Get Conditional Contributing parameters input
    with input_sidebar.expander("Input for DataFrame Conditional Contribution"):
        parameter_name_options = ["CPI change", "Fuel Escalation", "Electrical Escalation", "FX"]  # Add your parameters
        parameter_name = st.sidebar.selectbox("Select parameter name:", parameter_name_options, key='parameter_name')
        parameter_start_month = st.sidebar.text_input("Enter start month for parameter (MM/YYYY):", key='parameter_start_month')
        parameter_value = st.sidebar.number_input("Enter parameter value:", key='parameter_value')
        parameter_frequency_options = ["Monthly", "Quarterly", "Bi-Annual", "Annual"]
        parameter_frequency = st.sidebar.selectbox("Select parameter frequency:", parameter_frequency_options, key='parameter_frequency')
        # Display a "Save Changes" button
        save_changes = st.button("Save Conditional Parameter")

        # Append the parameter to the DataFrame when "Save Changes" is clicked
        if save_changes:
            new_parameter = {'Parameter': parameter_name,'Start Month': parameter_start_month, 
                             'Value': parameter_value, 'Escalation Frequency': parameter_frequency }
            st.session_state.parameters_df = pd.concat([st.session_state.parameters_df, pd.DataFrame([new_parameter])], ignore_index=True)
    
    # Get Base Lease parameters input                                                   
    with input_sidebar.expander("Input for Base Lease Changes"):
        location_name_options = ['Colocation R1 Grid', 'Colocation R2 Grid', 'Colocation R3 Grid', 
                                'Colocation R1 Off Grid', 'Colocation R2 Off Grid', 'Colocation R3 Off Grid']
        location_name = st.sidebar.selectbox("Select Location:", location_name_options, key='location_name')
        parameter_change_month = st.sidebar.text_input("Enter start month for parameter (MM/YYYY):", key='parameter_change_month')
        parameter_baselease_value = st.sidebar.number_input("Enter Base Lease value:", key='parameter_baselease_value')
        parameter_addon_value = st.sidebar.number_input("Enter Addon value:", key='parameter_addon_value')
        # Display a "Save Changes" button
        save_changes = st.button("Save Base Lease Changes")

        # Append the parameter to the DataFrame when "Save Changes" is clicked
        if save_changes:
            baselease_parameter = {'Location': location_name,'Change Months': parameter_change_month, 
                             'Base Rate': parameter_baselease_value, 'Addon Rate': parameter_addon_value }
            st.session_state.parameters_df2 = pd.concat([st.session_state.parameters_df2, pd.DataFrame([baselease_parameter])], ignore_index=True)


    # Right Main Section
    result_container = st.container()
    col1, col2 = st.columns(2)
    with col1:
        result_container.write("Contributional Contributional Parameters")
        result_container.write(st.session_state.parameters_df)
    with col2:
        result_container.write("Base Lease & AddOn Rates")
        result_container.write(st.session_state.parameters_df2)
                                                       
                                                       
    # Display the parameters table on the right side
    # Add logic to run simulation when "Simulate" is clicked
    simulate = result_container.button("Visualize Forecast")
    if simulate:
        result_container.header("Forecast Results")
        result_container.write(f"Performing forecast for {forecast_duration} months starting from {start_month}")
        
        
        # Run the Simulations
        visua, base, plot_col = forecast_logic(st.session_state.parameters_df, start_month, forecast_duration,
                                               st.session_state.parameters_df2)
        print("forecast R*C is ", visua.shape, "\n", visua.columns, "\n")
        visua_grp, visua_grp_perc = grouped_dwnld_excel(visua, grouping_frequency)
        print(visua_grp.shape, "and grouped is ", visua_grp_perc.shape, "\n")
        
        
        # create a excel writer object
        with pd.ExcelWriter(f"visua_allData_iter{round(iternum,0)}.xlsx") as writer:
            st.session_state.parameters_df.to_excel(writer, sheet_name="conditional params", index=False)
            st.session_state.parameters_df2.to_excel(writer, sheet_name="base lease params", index=False)
            base.to_excel(writer, sheet_name="base", index=False)
            visua.to_excel(writer, sheet_name="visua", index=False)
            visua_grp.to_excel(writer, sheet_name="visua_grouped", index=False)
            visua_grp_perc.to_excel(writer, sheet_name="visua_grouped_perc", index=False)
            

             
        # Plots
        plot_names = ['Grid Locations Revenue', 'OffGrid Locations Revenue', 'Total Revenue', 
                      'Escalations', 'Base Rate Variations', 'AddOn Variations']

        fig = make_subplots(rows = 3, cols = 2, subplot_titles = [f"Plot for {i}" for i in plot_names])
        for i, set_columns in enumerate(plot_col):
            row = i // 2 + 1
            col = i % 2 + 1
            for col_name in set_columns:
                fig.add_trace(  go.Scatter(x = visua['months'], y = visua[col_name], mode = 'lines+markers', name = col_name),
                                row = row, col= col)
                fig.update_xaxes(tickangle = -90, showgrid = True, gridcolor = 'lightgray', row = row, col = col)
                fig.update_yaxes(showgrid = False, row = row, col = col)
        fig.update_layout( height = 1200, showlegend=True, plot_bgcolor='white',
                          legend = dict(orientation='h', x=0, y=-0.25, traceorder='normal'),
                          margin = dict(l=10, r=10, t=70, b=70))

        
       
        
        chart = result_container.button("Show Graph")
#         st.plotly_chart(fig)
        if chart:
            st.plotly_chart(fig)
            
        table = result_container.button("Show Table")
#         st.table(visua_grp)
        if table:
            st.table(rev)
            
        
        # Download the table as a CSV file
        download_button = st.button("Download XLSX")
        if download_button:
            with pd.ExcelWriter(f"visua_allData_iter{round(iternum,0)}.xlsx") as writer:
                st.session_state.parameters_df.to_excel(writer, sheet_name="conditional params", index=False)
                st.session_state.parameters_df2.to_excel(writer, sheet_name="base lease params", index=False)
                base.to_excel(writer, sheet_name="base", index=False)
                visua.to_excel(writer, sheet_name="visua", index=False)
                visua_grp.to_excel(writer, sheet_name="visua_grouped", index=False)
                visua_grp_perc.to_excel(writer, sheet_name="visua_grouped_perc", index=False)

            



if __name__ == "__main__":
    main()
