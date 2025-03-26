
from data_prep import data_processing
from model_fitting import convol_neural
from data_upload import data_upload



if __name__ == "__main__":
    signal = data_processing('/home/tranzmeo/Documents/ModelData')
    signal.to_csv('signal.csv', index = False)
    data_upload(signal)
    convol_neural()

    