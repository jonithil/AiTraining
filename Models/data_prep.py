
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def index_filtering(data):
    
    starting_index = int(input("Give me the starting index to fetch data"))
    ending_index = int(input("Give me the ending index to fetch data"))
    selected_indices_df = pd.DataFrame(data.iloc[:, starting_index:ending_index+1])
    return selected_indices_df, starting_index, ending_index
    




def signal_identifier(starting_index, ending_index, data, selected_indices_df, save_dir="output_images"):

    os.makedirs(save_dir, exist_ok=True)

    for i in range(starting_index, ending_index+1):
        plt.figure(figsize=(10, 8))
        plt.plot(data['time'], selected_indices_df[str(i)])
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator())
        plt.xticks(rotation=45)

    
        save_path = os.path.join(save_dir, f"signal_{i}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")


def slicing(data, selected_indices_df):
    starting_time = input("Give me the starting time in %H-%M-%S format to fetch data")
    ending_time = input("Give me the ending time in %H-%M-%S format to fetch data")
    time_data = pd.DataFrame(data[['time']])
    filtered_data = pd.concat([selected_indices_df, time_data], axis=1)
    start = datetime.strptime(starting_time, '%H:%M:%S').time()
    end = datetime.strptime(ending_time, '%H:%M:%S').time()
    data['time'] = pd.to_datetime(data['time'])
    data['time_comp'] = data['time'].dt.time
    selected_time_df = pd.DataFrame(filtered_data.loc[ (data['time_comp'] >= start) & (data['time_comp']<= end) ])
    return selected_time_df
    


def interpolating(df):

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    df.set_index('time', inplace=True)

    df = df.ffill().bfill()

    df_interpolated = df.resample('100ms').asfreq()

    numeric_cols = df.select_dtypes(include=['number']).columns
    df_interpolated[numeric_cols] = df_interpolated[numeric_cols].interpolate(method='linear')

    df_interpolated = df_interpolated.ffill().bfill()

    df_interpolated.reset_index(drop = True, inplace=True)

    return df_interpolated



def reshaping(df):
    row_length, col_length = df.shape
    if row_length%100 != 0:
        row_tobe_deleted = row_length%100
        df.drop(df.tail(row_tobe_deleted).index, inplace=True)
    

    fixed_columns = 100
    output_data = []
    row_length = df.shape[0]

    for i in range(0, row_length, 100):
        rows_100 = df.iloc[i:i+100, :].values
        transposed_chunk = rows_100.T 
        output_data.append(transposed_chunk)  


    transposed_array = np.vstack(output_data)  

    transposed_df = pd.DataFrame(transposed_array)
    return transposed_df



def hdfs_loc_finder():
    pids_name = input("Please give the pids name")
    pids_number = input("Please give me pids number")
    hdfs_location = pids_name + '_' + pids_number 
    return hdfs_location


def typeIdentifier(hdfs_loc):
    simulation_log = pd.read_csv(r"/home/tranzmeo/Documents/MHMBPL simulation Final Log.csv")
    for index,row in simulation_log.iterrows():
        if row['HDFS Location'] == hdfs_loc:
            print(row['Activity'])
            eventType = row['Activity']
            return eventType
        



def data_processing(folder):
    files = natsorted(glob.glob(folder + '/*.csv'))
    signal = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file)
        print(f"Processing: {file}, Rows: {len(df)}")

        selected_index_data, starting_index, ending_index = index_filtering(df)
        signal_identifier(starting_index, ending_index, df, selected_index_data)
        selected_time_data = slicing(df, selected_index_data)
        print(f"After Slicing:{len(selected_time_data)}")

        df_interpolated = interpolating(selected_time_data)
        print(f"After Interpolation: {len(df_interpolated)}")

        transposed_df = reshaping(df_interpolated).copy()
        print(f"After Reshaping: {transposed_df.shape}")

        hdfs_location = hdfs_loc_finder()
        event_type = typeIdentifier(hdfs_location)
        transposed_df['eventtype'] = event_type

        signal = pd.concat([signal, transposed_df], axis=0, ignore_index=True)

    print(f"Final Signal Shape: {signal.shape}")

    return signal







    
