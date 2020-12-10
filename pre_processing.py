import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from functools import reduce

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
    # target_df.loc[:, 'Target_ID'] = vehicle_id
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
    # target_df.insert(0, "Target_ID", vehicle_id, True)
    target_df = target_df.reset_index(drop=True)

    return target_df

# Geração de um Dataframe com os veículos á frente ou atrás do veículo alvo passado
def get_preceeding_or_following_dataframe(full_df, target_df, position="preceeding"):
    target_vehicles_ids = target_df.Vehicle_ID.unique()
    # print(target_vehicles_ids)


    # if beside is True:
    #     main_vehicle_id = target_df.Target_ID.unique()
    #     print(main_vehicle_id)

    df_list = []
    for vehicle_id in target_vehicles_ids:
        aux_target_df = target_df.loc[target_df['Vehicle_ID'] == vehicle_id]
        frames_list = aux_target_df.Frame_ID.unique()
        aux_df = full_df.loc[full_df['Frame_ID'].isin(frames_list)]
        if position == "preceeding":
            aux_df = aux_df.loc[aux_df['Preceeding'] == vehicle_id]
        elif position == "following":
            aux_df = aux_df.loc[aux_df['Following'] == vehicle_id]
        # if beside is False:
        #     aux_df['Target_ID'] = vehicle_id
        # if beside is True:
        #     aux_df['Target_ID'] = main_vehicle_id[0]
        df_list.append(aux_df)

    if df_list:
        final_df = pd.concat(df_list)
        final_df = final_df[['Vehicle_ID',
                            # 'Target_ID',
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
        final_df = final_df.sort_values(by='Frame_ID')
        final_df = final_df.reset_index(drop=True)
    else:
        return pd.DataFrame()

    return final_df

# Adição de colunas das posições local e longitudinal com filtro Savitzky-Golay aplicado
def application_of_savgol_filter_and_return_dataframe(actual_df):
    if len(actual_df) == 0:
        return actual_df
    
    local_x = np.array(actual_df['Local_X_meters'])
    if local_x.all() >= 11:
        local_x_savgol = savgol_filter(local_x, 11, 2)
    else:
        local_x_savgol = local_x
    
    local_y = np.array(actual_df['Local_Y_meters'])
    if local_y.all() >= 11:
        local_y_savgol = savgol_filter(local_y, 11, 2)
    else:
        local_y_savgol = local_y
    
    actual_df['Local_X_filtered'] = pd.Series(local_x_savgol, index=actual_df.index)
    actual_df['Local_Y_filtered'] = pd.Series(local_y_savgol, index=actual_df.index)
        
    return actual_df


# Criação de Dataset com informações dos veículos à esquerda ou à direita do veículo alvo
def get_left_or_right_vehicle_dataframe(full_df, target_df, side="left"):
    frame_ids = target_df.Frame_ID.unique()
    target_vehicle_ids = target_df.Vehicle_ID.unique()
    
    side_df = full_df.loc[full_df['Frame_ID'].isin(frame_ids)]
    
    df_list = []
    for line in target_df.itertuples():
        if side == "left":
            left_lane = line.Lane_ID - 1
            aux_df = side_df.loc[side_df['Frame_ID'] == line.Frame_ID]
#             side_df = side_df.loc[(side_df['Vehicle_ID'] != line.Vehicle_ID) & (side_df['Local_X_meters'] < line.Local_X_meters) & (side_df['Lane_ID'] == (line.Lane_ID-1)) & (abs(side_df['Local_Y_meters'] - line.Local_Y_meters) < 1)]
            aux_df = aux_df.loc[side_df['Vehicle_ID'] != line.Vehicle_ID]
            aux_df = aux_df.loc[side_df['Local_X_meters'] < line.Local_X_meters]
            aux_df = aux_df.loc[side_df['Lane_ID'] == left_lane]
            aux_df = aux_df.loc[abs(side_df['Local_Y_meters'] - line.Local_Y_meters) < 1]
        if side == "right":
            right_lane = line.Lane_ID + 1
            aux_df = side_df.loc[side_df['Frame_ID'] == line.Frame_ID]
#             side_df = side_df.loc[(side_df['Vehicle_ID'] != line.Vehicle_ID) & (side_df['Local_X_meters'] < line.Local_X_meters) & (side_df['Lane_ID'] == (line.Lane_ID-1)) & (abs(side_df['Local_Y_meters'] - line.Local_Y_meters) < 1)]
            aux_df = aux_df.loc[side_df['Vehicle_ID'] != line.Vehicle_ID]
            aux_df = aux_df.loc[side_df['Local_X_meters'] > line.Local_X_meters]
            aux_df = aux_df.loc[side_df['Lane_ID'] == right_lane]
            aux_df = aux_df.loc[abs(side_df['Local_Y_meters'] - line.Local_Y_meters) < 1]
        # aux_df['Target_ID'] = line.Vehicle_ID
        df_list.append(aux_df)
    
    side_df = pd.concat(df_list)
    side_df = side_df[['Vehicle_ID',
                    #    'Target_ID',
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
    side_df = side_df.sort_values(by=['Vehicle_ID', 'Frame_ID'])
    side_df = side_df.reset_index(drop=True)

    return side_df

# Criação de Dataset com as informações finais do veículo alvo
def get_final_target_dataframe(target_df, df_type=0):
    # new_df = pd.DataFrame(columns=['Vehicle_ID', 
    #                                'Frame_ID',
    #                                'Global_Time',
    #                                'Local_X',
    #                                'Local_Y',
    #                                'Local_X_meters',
    #                                'Local_Y_meters',
    #                                'Local_X_Vel',
    #                                'Local_Y_Vel',
    #                                'v_Class'])

    local_x_vel_list = []
    local_y_vel_list = []
    have_velocities = []
    start_index = target_df.index.start
    for row in target_df.itertuples():
        if row.Index > start_index and row.Vehicle_ID == target_df['Vehicle_ID'][row.Index-1]:
            time = (row.Global_Time - target_df['Global_Time'][row.Index-1]) / 1000
            local_x_vel = (row.Local_X_meters - target_df['Local_X_meters'][row.Index-1]) / time
            local_y_vel = (row.Local_Y_meters - target_df['Local_Y_meters'][row.Index-1]) / time
            local_x_vel_list.append(local_x_vel)
            local_y_vel_list.append(local_y_vel)
            have_velocities.append(True)
        else:
            local_x_vel_list.append(0)
            local_y_vel_list.append(0)
            have_velocities.append(False)
            
            
    target_df['Local_X_Vel'] = pd.Series(local_x_vel_list, index=target_df.index)
    target_df['Local_Y_Vel'] = pd.Series(local_y_vel_list, index=target_df.index)
    target_df['Have_Velocities'] = pd.Series(have_velocities, index=target_df.index)
    target_df = target_df.loc[target_df['Have_Velocities'] != False]
    target_df = target_df.reset_index(drop=True)
    
    if df_type == 1:
        new_df = target_df[[
                            # 'Target_ID',
                        'Frame_ID', 
                        #    'Global_Time', 
                        'Local_X_meters',
                        'Local_Y_meters',
                        'Local_X_Vel',
                        'Local_Y_Vel']]
    else:
        new_df = target_df[[
                        # 'Target_ID',
                       'Frame_ID', 
                    #    'Global_Time', 
                       'Local_X_meters',
                       'Local_Y_meters',
                       'Local_X_Vel',
                       'Local_Y_Vel',
                       'v_Class']]
    
    # new_df = new_df.astype({'Vehicle_ID': 'int', 
    #                         'Frame_ID': 'int',
    #                         'Global_Time': 'int',
    #                         'v_Class': 'int'})
        
    return new_df

# Criação de Dataset com as informações finais dos veículos ao redor do alvo
def get_final_surround_dataframe(surround_df, target_df, df_type=0):
  
    if surround_df.empty and df_type == 1:
        return pd.DataFrame(columns=['Frame_ID',
                                   'Local_X_Vel',
                                   'Relative_Y_Vel',
                                   'Relative_Targ_X',
                                   'Relative_Targ_Y', 
                                   'Time_To_Collision'])
    elif surround_df.empty:
        return pd.DataFrame(columns=['Frame_ID',
                                   'Local_X_Vel',
                                   'Relative_Y_Vel',
                                   'Relative_Targ_X',
                                   'Relative_Targ_Y', 
                                   'Time_To_Collision',
                                   'v_Class'])
    
    local_x_vel_list = []
    relative_y_vel_list = []
    relative_local_x_targ_list = []
    relative_local_y_targ_list = []
    ttc_list = []
    have_velocities = []
    
    start_index = surround_df.index.start
    for row in surround_df.itertuples():
        if row.Index > start_index and row.Vehicle_ID == surround_df['Vehicle_ID'][row.Index-1]:
            time = (row.Global_Time - surround_df['Global_Time'][row.Index-1]) / 1000
            local_x_vel = (row.Local_X_meters - surround_df['Local_X_meters'][row.Index-1]) / time
            local_y_vel = (row.Local_Y_meters - surround_df['Local_Y_meters'][row.Index-1]) / time

            local_x_vel_list.append(local_x_vel)
            have_velocities.append(True)
            for line in target_df.itertuples():
                if row.Frame_ID == line.Frame_ID:
                    relative_y_vel = line.Local_Y_Vel - local_y_vel
                    local_x_targ = row.Local_X_meters - line.Local_X_meters
                    local_y_targ = row.Local_Y_meters - line.Local_Y_meters
                    ttc = local_y_targ / relative_y_vel
                    if math.isinf(ttc):
                        ttc = 0

                    relative_y_vel_list.append(relative_y_vel)
                    relative_local_x_targ_list.append(local_x_targ)
                    relative_local_y_targ_list.append(local_y_targ)
                    ttc_list.append(ttc)
                    break
        else:
            local_x_vel_list.append(0)
            relative_y_vel_list.append(0)
            relative_local_x_targ_list.append(0)
            relative_local_y_targ_list.append(0)
            ttc_list.append(0)
            have_velocities.append(False)
        
    surround_df['Local_X_Vel'] = pd.Series(local_x_vel_list, index=surround_df.index)
    surround_df['Relative_Y_Vel'] = pd.Series(relative_y_vel_list, index=surround_df.index)
    surround_df['Relative_Targ_X'] = pd.Series(relative_local_x_targ_list, index=surround_df.index)
    surround_df['Relative_Targ_Y'] = pd.Series(relative_local_y_targ_list, index=surround_df.index)
    surround_df['Time_To_Collision'] = pd.Series(ttc_list, index=surround_df.index)
    surround_df['Have_Velocities'] = pd.Series(have_velocities, index=surround_df.index)
    surround_df = surround_df.loc[surround_df['Have_Velocities'] != False]
    surround_df = surround_df.reset_index(drop=True)

    if df_type == 1:
        new_df = surround_df[[
                            #    'Vehicle_ID',
                            #    'Target_ID',
                            'Frame_ID',
                            'Local_X_Vel',
                            'Relative_Y_Vel',
                            'Relative_Targ_X',
                            'Relative_Targ_Y', 
                            'Time_To_Collision']]
    else:
        new_df = surround_df[[
                            #    'Vehicle_ID',
                            #    'Target_ID',
                            'Frame_ID',
                            'Local_X_Vel',
                            'Relative_Y_Vel',
                            'Relative_Targ_X',
                            'Relative_Targ_Y', 
                            'Time_To_Collision',
                            'v_Class']]
    
#     new_df = surround_df[df_list]

    # new_df = new_df.astype({'Frame_ID': 'int',
    #                         'v_Class': 'int'})
        
    return new_df

def get_all_targets_df(df):
    target_df = pd.DataFrame()
    
    ids_list = df.Vehicle_ID.unique()
    
    for vehicle_id in ids_list:
        aux_df = get_target_dataframe(df, vehicle_id)
        target_df = target_df.append(aux_df)
        
    target_df = target_df.dropna()
    
    return target_df

def get_complete_df(df, target_id, df_type=0):
    
    target_df = get_target_dataframe(df, target_id)
    target_df = application_of_savgol_filter_and_return_dataframe(target_df)
    behind_df = get_preceeding_or_following_dataframe(df, target_df)
    behind_df = application_of_savgol_filter_and_return_dataframe(behind_df)
    front_df = get_preceeding_or_following_dataframe(df, target_df, 'following')
    front_df = application_of_savgol_filter_and_return_dataframe(front_df)
    front_front_df = get_preceeding_or_following_dataframe(df, front_df, 'following')
    front_front_df = application_of_savgol_filter_and_return_dataframe(front_front_df)
    left_df = get_left_or_right_vehicle_dataframe(df, target_df)
    left_df = application_of_savgol_filter_and_return_dataframe(left_df)
    behind_left_df = get_preceeding_or_following_dataframe(df, left_df)
    behind_left_df = application_of_savgol_filter_and_return_dataframe(behind_left_df)
    front_left_df = get_preceeding_or_following_dataframe(df, left_df, 'following')
    front_left_df = application_of_savgol_filter_and_return_dataframe(front_left_df)
    right_df = get_left_or_right_vehicle_dataframe(df, target_df, 'right')
    right_df = application_of_savgol_filter_and_return_dataframe(right_df)
    behind_right_df = get_preceeding_or_following_dataframe(df, right_df)
    behind_right_df = application_of_savgol_filter_and_return_dataframe(behind_right_df)
    front_right_df = get_preceeding_or_following_dataframe(df, right_df, 'following')
    front_right_df = application_of_savgol_filter_and_return_dataframe(front_right_df)
    
    if df_type == 1:
        target_df = get_final_target_dataframe(target_df, 1)
        behind_df = get_final_surround_dataframe(behind_df, target_df, 1)
        front_df = get_final_surround_dataframe(front_df, target_df, 1)
        front_front_df = get_final_surround_dataframe(front_front_df, target_df, 1)
        left_df = get_final_surround_dataframe(left_df, target_df, 1)
        behind_left_df = get_final_surround_dataframe(behind_left_df, target_df, 1)
        front_left_df = get_final_surround_dataframe(front_left_df, target_df, 1)
        right_df = get_final_surround_dataframe(right_df, target_df, 1)
        behind_right_df = get_final_surround_dataframe(behind_right_df, target_df, 1)
        front_right_df = get_final_surround_dataframe(front_right_df, target_df, 1)
    else:
        target_df = get_final_target_dataframe(target_df)
        behind_df = get_final_surround_dataframe(behind_df, target_df)
        front_df = get_final_surround_dataframe(front_df, target_df)
        front_front_df = get_final_surround_dataframe(front_front_df, target_df)
        left_df = get_final_surround_dataframe(left_df, target_df)
        behind_left_df = get_final_surround_dataframe(behind_left_df, target_df)
        front_left_df = get_final_surround_dataframe(front_left_df, target_df)
        right_df = get_final_surround_dataframe(right_df, target_df)
        behind_right_df = get_final_surround_dataframe(behind_right_df, target_df)
        front_right_df = get_final_surround_dataframe(front_right_df, target_df)
    
    target_df.columns = target_df.columns.map(lambda x: str(x) + '_targ' if str(x) != 'Frame_ID' else str(x))
    behind_df.columns = behind_df.columns.map(lambda x: str(x) + '_b' if str(x) != 'Frame_ID' else str(x))
    front_df.columns = front_df.columns.map(lambda x: str(x) + '_f' if str(x) != 'Frame_ID' else str(x))
    front_front_df.columns = front_front_df.columns.map(lambda x: str(x) + '_ff' if str(x) != 'Frame_ID' else str(x))
    left_df.columns = left_df.columns.map(lambda x: str(x) + '_l' if str(x) != 'Frame_ID' else str(x))
    behind_left_df.columns = behind_left_df.columns.map(lambda x: str(x) + '_bl' if str(x) != 'Frame_ID' else str(x))
    front_left_df.columns = front_left_df.columns.map(lambda x: str(x) + '_fl' if str(x) != 'Frame_ID' else str(x))
    right_df.columns = right_df.columns.map(lambda x: str(x) + '_r' if str(x) != 'Frame_ID' else str(x))
    behind_right_df.columns = behind_right_df.columns.map(lambda x: str(x) + '_br' if str(x) != 'Frame_ID' else str(x))
    front_right_df.columns = front_right_df.columns.map(lambda x: str(x) + '_fr' if str(x) != 'Frame_ID' else str(x))

    dfs = [target_df, 
           behind_df, 
           front_df, 
           front_front_df, 
           left_df, 
           behind_left_df, 
           front_left_df, 
           right_df, 
           behind_right_df, 
           front_right_df]
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['Frame_ID'],
                                                   suffixes=(False, False),
                                                   how='outer'), dfs).fillna(0)
    
    return final_df