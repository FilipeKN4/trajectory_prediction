import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Adição de colunas das posições local e longitudinal dos veículos em metros
def add_positions_in_meters(full_df):
    local_x = []
    local_y = []
    for row in full_df.itertuples():
        local_x.append(row.Local_X / 3.2808)
        local_y.append(row.Local_Y / 3.2808)

    full_df['Local_X_meters'] = pd.Series(local_x, index=full_df.index)
    full_df['Local_Y_meters'] = pd.Series(local_y, index=full_df.index)

    return full_df

# Criação de dataframe com informações do veículo de interesse
def get_target_dataframe(full_df, vehicle_id):
    target_df = full_df.loc[full_df['Vehicle_ID'] == vehicle_id]
    target_df = target_df[['Vehicle_ID',
                           'Frame_ID', 
                           'Global_Time', 
                           'Local_X',
                           'Local_Y',
                           'Local_X_meters',
                           'Local_Y_meters',
                           'v_Vel',
                           'v_Class',
                           'Lane_ID',
                           'Preceeding',
                           'Following']]
    target_df = target_df.reset_index(drop=True)

    return target_df

# Geração de um Dataframe com os veículos á frente ou atrás do veículo alvo passado
def get_preceeding_or_following_dataframe(full_df, target_df, position="preceeding"):
    target_vehicles_ids = target_df.Vehicle_ID.unique()
    print(target_vehicles_ids)

    df_list = []
    for vehicle_id in target_vehicles_ids:
        aux_target_df = target_df.loc[target_df['Vehicle_ID'] == vehicle_id]
        frames_list = aux_target_df.Frame_ID.unique()
        aux_df = full_df.loc[full_df['Frame_ID'].isin(frames_list)]
        if position == "preceeding":
            aux_df = aux_df.loc[aux_df['Preceeding'] == vehicle_id]
        elif position == "following":
            aux_df = aux_df.loc[aux_df['Following'] == vehicle_id]
        df_list.append(aux_df)

    if df_list:
        final_df = pd.concat(df_list)
        final_df = final_df[['Vehicle_ID',
                            'Frame_ID', 
                            'Global_Time', 
                            'Local_X',
                            'Local_Y',
                            'Local_X_meters',
                            'Local_Y_meters',
                            'v_Vel',
                            'v_Class',
                            'Lane_ID',
                            'Preceeding',
                            'Following']]
        final_df = final_df.reset_index(drop=True)
    else:
        return df_list

    return final_df

# Adição de colunas das posições local e longitudinal com filtro Savitzky-Golay aplicado
def application_of_savgol_filter_and_return_dataframe(actual_df):
    if len(actual_df) == 0:
        return actual_df
    
    local_x = np.array(actual_df['Local_X_meters'])
    local_x_savgol = savgol_filter(local_x, 11, 2)
    
    local_y = np.array(actual_df['Local_Y_meters'])
    local_y_savgol = savgol_filter(local_y, 11, 2)
    
    actual_df['Local_X_filtered'] = pd.Series(local_x_savgol, index=actual_df.index)
    actual_df['Local_Y_filtered'] = pd.Series(local_y_savgol, index=actual_df.index)
        
    return actual_df


# Criação de Dataset com informações dos veículos à esquerda ou à direita do veículo alvo
def get_left_or_right_vehicle_dataframe(full_df, target_df, side="left"):   
    df_list = []
    for row in target_df.itertuples():
        aux_df = full_df.loc[full_df['Frame_ID'] == row.Frame_ID]
        df_list.append(aux_df.loc[aux_df['Vehicle_ID'] != row.Vehicle_ID])     
    side_df = pd.concat(df_list)

    side_df_index = []
    for row in side_df.itertuples():
        for line in target_df.itertuples():
            if row.Frame_ID == line.Frame_ID:
                if side == "left":
                    if row.Local_X < line.Local_X and row.Lane_ID == (line.Lane_ID-1) and abs(row.Local_Y_meters - line.Local_Y_meters) < 1:
                        side_df_index.append(row.Index)
                        break
                elif side == "right":
                    if row.Local_X > line.Local_X and row.Lane_ID == (line.Lane_ID+1) and abs(row.Local_Y - line.Local_Y) < 1:
                        side_df_index.append(row.Index)
                        break
                        
    side_df = side_df.loc[side_df.index.isin(side_df_index)]
    side_df = side_df[['Vehicle_ID',
                       'Frame_ID', 
                       'Global_Time', 
                       'Local_X',
                       'Local_Y',
                       'Local_X_meters',
                       'Local_Y_meters',
                       'v_Vel',
                       'v_Class',
                       'Lane_ID',
                       'Preceeding',
                       'Following']]
    side_df = side_df.reset_index(drop=True)

    return side_df