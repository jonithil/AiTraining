
from model import convol_neural
import pandas as pd



if __name__ == "__main__":
    #signal = data_processing('/home/tranzmeo/Documents/ModelData')
    signal = pd.read_csv(r'/home/tranzmeo/Learning/model creation/AiTraining/ModelData/INTERAMP_MHMBPL_TRAIN1.CSV', header = None)
    signal.drop(columns=signal.columns[-1],  axis=1,  inplace=True)
    #signal.to_csv('signal.csv', index = False)
    signal = signal.loc[signal.iloc[:, -1].isin(['Manual Digging', 'Machine Digging', 'Vehicle Movement'])]
    #data_upload(signal)
    convol_neural(signal)
    #recurrent(signal)




