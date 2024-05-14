import pandas as pd
import numpy as np 
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset

diagnosis_pipeline = load_pipeline(config.MODEL_NAME)

# For running pytest
def generate_predictions(data_input):
   data = pd.DataFrame(data_input)
   pred = diagnosis_pipeline.predict(data[config.FEATURES])
   output = np.where(pred==1,'M', 'B')
   result = {"prediction":output}
   return result

# for generating the prediction
# def generate_predictions():    
#     test_data = load_dataset(config.TEST_FILE)
#     pred = diagnosis_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1,'Y', 'N')
#     print(output)
#     #result = {"Predictions":output}
#     return output  

if __name__ =="__main__":
    generate_predictions()