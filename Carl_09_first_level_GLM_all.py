import pandas as pd
from Carl_Response_Functions.first_level_GLM import *

#TODO Modify the paths here to point to "non-discretized" responses
#TODO check the subject files df and the preprocessed_data is all there
#TODO check all the paths as usual:)
#TODO Then this can just be run via slurm
#TODO Change the slurm path so that it points to this not Maeghals
#just change this to modify the responses folder used by the code
path = "Carl_preprocessed_responses/"
subject_files_df = pd.read_excel("subject_files_feb_2023.xlsx",index_col=0)
preproc_path = "preprocessed_and_resampled_data/"
z_maps_path = "Carl_first_level_z_maps_resampled_mask_2"

run_first_level_glm(z_maps_path,preproc_path,subject_files_df,path)