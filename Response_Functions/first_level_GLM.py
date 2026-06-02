import os
import nibabel as nib
import nilearn
import numpy as np
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map
import pandas as pd
from nilearn.plotting import plot_glass_brain
import matplotlib.pyplot as plt
from Response_Functions.load_responses import readCsv



def get_mask_file(subject_files_df,response, selected_run):

  mask_file_sp = subject_files_df.loc[(
    subject_files_df.subject == int(response['subject'][4:])) & 
    (subject_files_df.session == int(response['visit'][-1]))
    ][f"sp_{selected_run}_brain_mask"].to_list()[0]
  print("mask file:", mask_file_sp)
  return mask_file_sp


def save_results(GLMPath, subject,z_map, z_map_pain, z_map_movement, Beta_concat, Beta_concat_pain, Beta_concat_movement):
  os.makedirs(GLMPath)
  stat_names = ["z_map", "z_map_pain", "z_map_movement", "Beta_concat", "Beta_concat_pain", "Betas_concat_movement"]
  for j, stat in enumerate([z_map, z_map_pain, z_map_movement, Beta_concat, Beta_concat_pain, Beta_concat_movement]):
    path = os.path.join(GLMPath,f'{stat_names[j]}.nii.gz')
    nib.save(stat, path)

  print(f"saved {str(list(subject))}")


def run_first_level_glm(z_maps_path,preproc_path,subject_files_df,path):
  """
  Runs the first level GLM for the chronic patients. Saves the results to `z_maps_path`. If the subjects have already been done, then says this, to prevent 
  overwriting. 
  """

  regressors = {
    'errors': readCsv("convolved_lag_1_differences.csv",path=path),
    'response': readCsv("convolved_responses.csv",path),
    'movement_regressor': readCsv("convolved_movement_regressors.csv",path),
  }

  slice_times = np.arange(regressors['response'].shape[1])*2.5 #should be 240
  if (len(slice_times) != 240):
    raise ValueError("The responses are the wrong length?")
  
  responses_details_df = pd.read_csv(os.path.join(path,"preprocessed_response_details.csv"), delimiter='\t')

  for df_index, subject in responses_details_df.iterrows():

    firstLevelGLMPath = os.path.join(z_maps_path, subject["subject"], "ses-"+subject["visit"])
    print(f"Running {str(list(subject))}")

    if os.path.isdir(firstLevelGLMPath):
      print("Already done this subject!")
      continue
      

    if int(subject['subject'][-3:]) == 98: #or int(subject['subject'][-3:]) > 97:
      print("not this subject 98")
      continue

    try:
      subject_fMRI_path = os.path.join(preproc_path,subject['subject'],f"ses-{subject['visit']}",subject["run"],'cleaned_file.nii.gz')
      resampled_subject_fMRI = nilearn.image.load_img(subject_fMRI_path)
    except Exception as e:
      print(f"the fMRI file for {str(list(subject))} is not available at {subject_fMRI_path}, because {e}. Continuing")
      continue

    if resampled_subject_fMRI.shape != (99, 117, 95, 240):
      print(resampled_subject_fMRI.shape)
      print("This is the wrong shape!!")
      continue

    try:
      mask_file_sp = get_mask_file(subject_files_df,subject)
    except:
      print(f"The mask file for  {list(subject)} is not available. Continuing")
      continue

    #the regressors should already be convolved, so I just need to create the design matrix.
    df_events = pd.DataFrame()
    df_events["onset"] = slice_times
    df_events['modulation'] = regressors['errors'][df_index]
    df_events["duration"] = 2.5
    df_events["trial_type"] = 'errors'

    stacked_regressors = np.vstack((regressors['response'][df_index],regressors['movement_regressor'][df_index])).T

    #make the design matrix
    design_matrix = make_first_level_design_matrix(
      frame_times = slice_times,
      events = df_events, 
      drift_model='polynomial', 
      drift_order=3, 
      hrf_model=None,
      add_regs= stacked_regressors,
      add_reg_names= ['response','movement_regressor']
    )

    #Create a first level model with 
    template = load_mni152_template()
    fmri_glm = FirstLevelModel(minimize_memory=True, mask_img=resample_to_img(mask_file_sp, template, interpolation='nearest'))
    fmri_glm = fmri_glm.fit(resampled_subject_fMRI, design_matrices=design_matrix)

    # Compute the error contrasts
    contrast_vector = np.array([1, 0, 0, 0, 0, 0, 0])
    effects_map = fmri_glm.compute_contrast(contrast_vector,output_type= 'effect_size')
    z_map = fmri_glm.compute_contrast(contrast_vector)

    # Compute the pain contrasts
    contrast_vector_2 = np.array([0, 1, 0, 0, 0, 0, 0])
    effects_map_pain = fmri_glm.compute_contrast(contrast_vector_2,output_type= 'effect_size')
    z_map_pain = fmri_glm.compute_contrast(contrast_vector_2)

    # Compute the movement contrasts
    contrast_vector_3 = np.array([0, 0, 1, 0, 0, 0, 0])
    effects_map_movement = fmri_glm.compute_contrast(contrast_vector_3,output_type= 'effect_size')
    z_map_movement = fmri_glm.compute_contrast(contrast_vector_3)
    
    save_results(firstLevelGLMPath,subject,z_map,z_map_pain, z_map_movement, effects_map,effects_map_pain, effects_map_movement)


def run_first_level_glm(z_maps_path,preproc_path,subject_files_df,path, regressors):
  """
  Runs the first level GLM for the chronic patients. Saves the results to `z_maps_path`. If the subjects have already been done, then says this, to prevent 
  overwriting.
  #TODO Add LOGIC which looks in the preproc_path for the latest run. The regressors will also need to be updated.
  """

  slice_times = np.arange(regressors['response'].shape[1])*2.5 #should be 240
  if (len(slice_times) != 240):
    raise ValueError("The responses are the wrong length?")

  #! This will have the run number in from now on, so you can select the response with the largest run from this.
  responses_details_df = pd.read_csv(os.path.join(path,"preprocessed_response_details.csv"), delimiter='\t')

  # Pick out highest run per patient and visit
  for subject_id in responses_details_df["subject"]:
      response = responses_details_df[responses_details_df["subject"] == subject_id]
      
      #Collect the visits per subject that have a run_02
      second_run = response[response["run"] == "run_02"]
      visits_with_second_run = second_run["visit"]

      #Check if the subject has a run_01 for the same visit for which they have a run_02 
      index_to_drop = response[(response["visit"].isin(visits_with_second_run)) & (response["run"] == "run_01")].index

      #Drop run_01 for those visits that had both run_01 and run_02
      responses_details_df = responses_details_df.drop(index_to_drop)

  for df_index, subject in responses_details_df.iterrows():

    firstLevelGLMPath = os.path.join(z_maps_path, subject["subject"], "ses-"+subject["visit"], subject["run"])
    print(f"Running {str(list(subject))}")

    if os.path.isdir(firstLevelGLMPath):
      print("Already done this subject!")
      continue
      
    if int(subject['subject'][-3:]) == 98: #or int(subject['subject'][-3:]) > 97:
      print("not this subject 98")
      continue

    #adusting name to "sp_run_01" or "sp_run_02" for path finding in preprocessed_and_resampled_data
    selected_run = "sp_" + subject["run"]
    chosen_run = subject["run"]

    try:
      #Updated regressors to include run02 in case a subject has both run01 and run02
      subject_fMRI_path = os.path.join(preproc_path,subject['subject'],f"ses-{subject['visit']}",selected_run,'cleaned_file.nii.gz')
      print(subject_fMRI_path)
      resampled_subject_fMRI = nilearn.image.load_img(subject_fMRI_path)
    except Exception as e:
      print(f"the fMRI file for {str(list(subject))} is not available at {subject_fMRI_path}, because {e}. Continuing")
      continue

    if resampled_subject_fMRI.shape != (99, 117, 95, 240):
      print(resampled_subject_fMRI.shape)
      print("This is the wrong shape!!")
      continue

    try:
      mask_file_sp = get_mask_file(subject_files_df,subject, selected_run = chosen_run)
    except Exception as e:
      print(f"The mask file for  {list(subject)} is not available because {e}. Continuing")
      continue

    #the regressors should already be convolved, so I just need to create the design matrix.
    df_events = pd.DataFrame()
    df_events["onset"] = slice_times
    df_events['modulation'] = regressors['errors'][df_index]
    df_events["duration"] = 2.5
    df_events["trial_type"] = 'errors'

    stacked_regressors = np.vstack((regressors['response'][df_index],regressors['movement_regressor'][df_index])).T

    #make the design matrix
    design_matrix = make_first_level_design_matrix(
      frame_times = slice_times,
      events = df_events, 
      drift_model='polynomial', 
      drift_order=3, 
      hrf_model=None,
      add_regs= stacked_regressors,
      add_reg_names= ['response','movement_regressor']
    )

    #Create a first level model with 
    template = load_mni152_template()
    fmri_glm = FirstLevelModel(minimize_memory=True, mask_img=resample_to_img(mask_file_sp, template, interpolation='nearest'))
    fmri_glm = fmri_glm.fit(resampled_subject_fMRI, design_matrices=design_matrix)

    # Compute the error contrasts
    contrast_vector = np.array([1, 0, 0, 0, 0, 0, 0])
    effects_map = fmri_glm.compute_contrast(contrast_vector,output_type= 'effect_size')
    z_map = fmri_glm.compute_contrast(contrast_vector)

    # Compute the pain contrasts
    contrast_vector_2 = np.array([0, 1, 0, 0, 0, 0, 0])
    effects_map_pain = fmri_glm.compute_contrast(contrast_vector_2,output_type= 'effect_size')
    z_map_pain = fmri_glm.compute_contrast(contrast_vector_2)

    # Compute the movement contrasts
    contrast_vector_3 = np.array([0, 0, 1, 0, 0, 0, 0])
    effects_map_movement = fmri_glm.compute_contrast(contrast_vector_3,output_type= 'effect_size')
    z_map_movement = fmri_glm.compute_contrast(contrast_vector_3)
    
    save_results(firstLevelGLMPath,subject,z_map,z_map_pain, z_map_movement, effects_map,effects_map_pain, effects_map_movement)


def plot_z_maps(maps,subjects_studied):
  for i,map in enumerate(maps):
    subject = subjects_studied[i]
    print(map.shape)
    plot_stat_map(map, title=f"z_stats for error contrast {list(subject)[0]}", threshold=1)


def plot_stat_map_from_path(path):
    img = nib.load(path)
    print(img.affine)
    plot_stat_map(img, threshold=1)

def plot_z_maps_from_directory(z_map_path,threshold=1,colorbar=True,map_type='z_map'):
    """ Takes a directory structure of subjects/visits/z_maps, and iterates through it. 
    `map_type` is the name of the type of z map in the directory, e.g. z_map_movement."""
    for subject_dir in sorted(os.listdir(z_map_path)):
        for visit in sorted(os.listdir(os.path.join(z_map_path, subject_dir))):
            path = os.path.join(z_map_path, subject_dir, visit, f'{map_type}.nii.gz')
            z_map = nib.load(path)
            plot_glass_brain(z_map,threshold=threshold,colorbar=colorbar)
            plt.show()


def plot_grid_of_first_level_GLMs(z_maps_path) -> "tuple[pd.DataFrame, list]":
  """
  Plots a grid showing which subjects now have first level GLMs, given the npz file with
  information about which subjects have had their first level GLMs done.

  Returns a pandas_df
  
  """
  first_level_z_map_files = []
  subjects = []
  for subject_dir in sorted(os.listdir(z_maps_path)):
    if subject_dir[:4] == 'sub-':
      subjects.append(subject_dir)
      for visit_dir in sorted(os.listdir(os.path.join(z_maps_path,subject_dir))):
        z_map_file = os.path.join(z_maps_path,subject_dir,visit_dir)
        first_level_z_map_files.append(z_map_file)

  N_subjects = len(subjects) 

  # Create a grid of visits
  visitsGrid = np.full((N_subjects,5),False)
  for i in range(N_subjects):
    visitsPresent = os.listdir(os.path.join(z_maps_path,subjects[i]))
    visitIds = [int(visit[-1]) -1 for visit in visitsPresent]
    # try:
    #   visitIds.remove(4)
    # except:
    #   pass
    visitsGrid[i,visitIds] = True
  

  fig, ax = plt.subplots(figsize=(15,4))
  plt.title("Complete First Level GLM data for All Subjects",fontsize=20)
  plt.imshow(visitsGrid.T,aspect='auto')
  _ = plt.ylabel("Visit",fontsize=20)
  _ = plt.xlabel("Subject",fontsize=20)
  plt.yticks(np.arange(5), np.arange(1,6))

  ax.set_xticks(np.arange(N_subjects)[::2])
  ax.set_xticklabels(subjects[::2], rotation=90)

  # create legend with custom handles and labels
  handles = []
  handles.append(plt.scatter([], [], color='purple', label='Data',edgecolors='black'))
  handles.append(plt.scatter([], [], color='yellow', label='No Data',edgecolors='black'))
  plt.legend(handles=handles, labels=['No Data','Data'], loc='upper left')
  plt.show()
  visitsDf = pd.DataFrame(visitsGrid,index=subjects,columns=[f'visit{i}' for i in [1,2,3,4,5]])
  return visitsDf, subjects