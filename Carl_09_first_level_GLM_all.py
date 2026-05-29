import pandas as pd
from Response_Functions.first_level_GLM import *

#just change this to modify the responses folder used by the code
path = "Carl_preprocessed_responses/"
subject_files_df = pd.read_excel("subject_files_feb_2023.xlsx",index_col=0)
preproc_path = "preprocessed_and_resampled_data/"
z_maps_path = "Carl_first_level_z_maps_resampled_mask_2"

run_first_level_glm(z_maps_path,preproc_path,subject_files_df,path)