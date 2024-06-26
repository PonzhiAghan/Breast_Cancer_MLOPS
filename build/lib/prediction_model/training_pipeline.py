import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,save_pipeline
from prediction_model.processing import preprocessing as pp
import prediction_model.pipeline as pipe
import sys

def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({'B':0,'M':1})
    pipe.diagnosis_pipeline.fit(train_data[config.FEATURES],train_y)
    save_pipeline(pipe.diagnosis_pipeline)

if __name__== '__main__':
    perform_training()