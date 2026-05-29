"""
Functions important for running the second level GLM on the responses.
"""

import numpy as np
import pandas as pd
import os
import nibabel as nib
from nilearn import image, datasets
from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.image import coord_transform, resample_to_img
from nilearn.plotting import plot_glass_brain

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def createFirstLevelResponsesDF(z_maps_path, participants_df, visitsGrid, recovering, persisting, run_01_df, run_02_df) -> pd.DataFrame:
  """
  Returns a pandas dataframe with columns `visit1` `visit2` `visit3` `visit4` `visit5` `age` `chronic` `SBPp` `SBPr`.

  Takes every subject in the first level `z_maps_path` directory as a dataframe row. 
  `participants_df` is a dataframe which must have a 'participant_id' and 'group' column.
  `visitsGrid` is a pandas df with columns 'visit1', etc where a value of True means a subject was present at that visit.

  Basically collates all the useful information about the subjects. 
  """
  subjects = []
  for subject_dir in sorted(os.listdir(z_maps_path)):
    if subject_dir[:4] == 'sub-':
      subjects.append(subject_dir)
  subjects = np.array(subjects)
  ages = list(participants_df.loc[participants_df['participant_id'].isin(subjects)]['age'])
  secondLevelGLMDF = pd.DataFrame(
    visitsGrid,
    index=subjects,
    columns=['visit1', 'visit2','visit3','visit4','visit5']
  ) # create the initial dataframe

  #now populate with ages
  secondLevelGLMDF['age'] = ages - np.mean(ages)

  #now add a column which is 1 if the subject is chronic
  subjects_groups = np.array(participants_df.loc[participants_df['participant_id'].isin(subjects)]['group'])
  chronic = np.where(subjects_groups == 'chronic',1,0)
  secondLevelGLMDF['chronic'] = chronic

  # now add a column which is 1 for the classification of the subject
  secondLevelGLMDF['SBPr'] = 0
  secondLevelGLMDF['SBPp'] = 0
  secondLevelGLMDF.loc[recovering, 'SBPr'] = 1
  secondLevelGLMDF.loc[persisting, 'SBPp'] = 1

#Adding run information to the output as we have both run_01 and run_02

  for i in range(1, 5):  # visits 1 to 4
    secondLevelGLMDF[f'run_01_visit{i}'] = 0
    secondLevelGLMDF[f'run_02_visit{i}'] = 0

  for _, row in run_01_df.iterrows():
    subj = row["subject"]
    visit = row["visit"]
    visit_col = f'run_01_{visit}' 
    if subj in secondLevelGLMDF.index and visit_col in secondLevelGLMDF.columns:
        secondLevelGLMDF.loc[subj, visit_col] = 1

  for _, row in run_02_df.iterrows():
    subj = row["subject"]
    visit = row["visit"]
    visit_col = f'run_02_{visit}' 
    if subj in secondLevelGLMDF.index and visit_col in secondLevelGLMDF.columns:
        secondLevelGLMDF.loc[subj, visit_col] = 1

  print(secondLevelGLMDF)

  return secondLevelGLMDF


def create_design_matrix_for_single_visit(z_maps_path,USE_Z_MAPS,secondLevelGLMDF,visit = 'visit1', type_of_image='error'):
  """Takes the visit as a string e.g. `"visit1"` """

  subjects_with_visit_df = secondLevelGLMDF.loc[secondLevelGLMDF[visit] == True][['SBPr', 'SBPp', 'chronic', 'age']]
  matrix = subjects_with_visit_df.sort_values(by=['SBPr', 'SBPp', 'chronic'], ascending=False)
  matrix['niftiImage'] = 0

  map_type = "z_map" if USE_Z_MAPS else "Beta_concat"
  first_level_file = f'{map_type}.nii.gz' if type_of_image == "error" else f'{map_type}_pain.nii.gz'
  for subject, information in matrix.iterrows():

    if secondLevelGLMDF.loc[subject, f'run_01_{visit}'] == 1:
        run_type_visit = "run_01"
    elif secondLevelGLMDF.loc[subject, f'run_02_{visit}'] == 1:
        run_type_visit = "run_02"

    z_map = nib.load(os.path.join(z_maps_path, subject, f'ses-{visit}', run_type_visit, first_level_file))
    matrix.loc[subject,'niftiImage'] = z_map

  design_matrix = matrix.loc[:, matrix.columns != 'niftiImage']
  niftiImages = list(matrix['niftiImage'])

  return design_matrix, niftiImages

def create_design_matrix_across_visits(z_maps_path, USE_Z_MAPS,secondLevelGLMDF,first_visit = 'visit1', second_visit='visit4', group='SBPr', type_of_image='error'):
  """Takes the visit as a string e.g. `"visit1"` """

  subjects_with_visit_df = secondLevelGLMDF.loc[(secondLevelGLMDF[first_visit] == True) & (secondLevelGLMDF[second_visit] == True) &  (secondLevelGLMDF[group] == 1)]
  subjects_with_visit_df = subjects_with_visit_df[[group, 'age']]

  subjects_with_visit_df['niftiImage'] = 0
  matrix = pd.DataFrame(columns=['subject', first_visit,second_visit,'age', 'niftiImage'])

  map_type = "z_map" if USE_Z_MAPS else "Beta_concat"
  first_level_file = f'{map_type}.nii.gz' if type_of_image == "error" else f'{map_type}_pain.nii.gz'
  for subject, information in subjects_with_visit_df.iterrows():

    if secondLevelGLMDF.loc[subject, f'run_01_{first_visit}'] == 1:
        run_type_first = "run_01"
    elif secondLevelGLMDF.loc[subject, f'run_02_{first_visit}'] == 1:
        run_type_first = "run_02"
  
    if secondLevelGLMDF.loc[subject, f'run_01_{second_visit}'] == 1:
        run_type_second = "run_01"
    elif secondLevelGLMDF.loc[subject, f'run_02_{second_visit}'] == 1:
        run_type_second = "run_02"

    z_map_first = nib.load(os.path.join(z_maps_path, subject, f'ses-{first_visit}', run_type_first, first_level_file))
    z_map_second = nib.load(os.path.join(z_maps_path, subject, f'ses-{second_visit}',run_type_second, first_level_file))

    # Create new rows
    row1 = pd.Series({'subject': subject, first_visit: 1, second_visit: 0, 'age': information['age'], 'niftiImage': z_map_first})
    row2 = pd.Series({'subject': subject, first_visit: 0, second_visit: 1, 'age': information['age'], 'niftiImage': z_map_second})
    matrix = matrix.append([row1, row2], ignore_index=True)

  design_matrix = matrix[[first_visit,second_visit,'age']].astype(float)

  return design_matrix, list(matrix['niftiImage'])

def create_movement_design_matrix(z_maps_path, USE_Z_MAPS, secondLevelGLMDF):
  subjects_with_visit_df = secondLevelGLMDF.loc[(secondLevelGLMDF['visit1'] == True) & (secondLevelGLMDF['visit4'] == True)]
  #subjects_with_visit_df = subjects_with_visit_df[['age']]
  
  matrix = pd.DataFrame(columns=['subject', 'age', 'constant', 'niftiImage'])

  map_type = "z_map" if USE_Z_MAPS else "Betas_concat"
  first_level_file = f'{map_type}_movement.nii.gz'

  for subject, information in subjects_with_visit_df.iterrows():

    if secondLevelGLMDF.loc[subject, 'run_01_visit1'] == 1:
        run_type_visit1 = "run_01"
    elif secondLevelGLMDF.loc[subject, 'run_02_visit1'] == 1 == 1:
        run_type_visit1 = "run_02"
  
    if secondLevelGLMDF.loc[subject, 'run_01_visit4'] == 1:
        run_type_visit4 = "run_01"
    elif secondLevelGLMDF.loc[subject, 'run_02_visit4'] == 1 == 1:
        run_type_visit4 = "run_02"


    z_map_first = nib.load(os.path.join(z_maps_path, subject, f'ses-visit1', run_type_visit1, first_level_file))
    z_map_second = nib.load(os.path.join(z_maps_path, subject, f'ses-visit4', run_type_visit4,  first_level_file))

    # Create new rows
    row1 = pd.Series({'subject': subject, 'age': information['age'], 'constant': 1, 'niftiImage': z_map_first})
    row2 = pd.Series({'subject': subject, 'age': information['age'], 'constant': 1, 'niftiImage': z_map_second})
    matrix = matrix.append([row1, row2], ignore_index=True)

  design_matrix = matrix[['age', 'constant']].astype(float)

  return design_matrix, list(matrix['niftiImage'])


def create_design_matrix_all_chronic(z_maps_path, USE_Z_MAPS, secondLevelGLMDF,type_of_image):
  """ Creates a design matrix for the chronic patients in visits one and four"""
  subjects_with_visit_df = secondLevelGLMDF.loc[(secondLevelGLMDF['visit1'] == True) & (secondLevelGLMDF['visit4'] == True)]
  subjects_with_visit_df = subjects_with_visit_df[['age']]
  matrix = pd.DataFrame(columns=['subject', 'age', 'constant', 'niftiImage'])

  map_type = "z_map" if USE_Z_MAPS else "Beta_concat"
  first_level_file = f'{map_type}.nii.gz' if type_of_image == "error" else f'{map_type}_pain.nii.gz'
  for subject, information in subjects_with_visit_df.iterrows():

    if secondLevelGLMDF.loc[subject, 'run_01_visit1'] == 1:
        run_type_visit1 = "run_01"
    elif secondLevelGLMDF.loc[subject, 'run_02_visit1'] == 1 == 1:
        run_type_visit1 = "run_02"
  
    if secondLevelGLMDF.loc[subject, 'run_01_visit4'] == 1:
        run_type_visit4 = "run_01"
    elif secondLevelGLMDF.loc[subject, 'run_02_visit4'] == 1 == 1:
        run_type_visit4 = "run_02"
    
    z_map_first = nib.load(os.path.join(z_maps_path, subject, f'ses-visit1', run_type_visit1, first_level_file))
    z_map_second = nib.load(os.path.join(z_maps_path, subject, f'ses-visit4', run_type_visit4,  first_level_file))

    # Create new rows
    row1 = pd.Series({'subject': subject, 'age': information['age'], 'constant': 1, 'niftiImage': z_map_first})
    row2 = pd.Series({'subject': subject, 'age': information['age'], 'constant': 1, 'niftiImage': z_map_second})
    matrix = matrix.append([row1, row2], ignore_index=True)

  design_matrix = matrix[['age', 'constant']].astype(float)

  return design_matrix, list(matrix['niftiImage'])


def run_second_level_GLM(design_matrix, niftiImages,  contrast, save_name):
  """Computes a second level GLM, and plots the corresponding z_stat_map for the contrast given.
  Contrasts are given as strings e.g. `"visit1"`, `"SBPr - chronic"`  """

  model = SecondLevelModel(mask_img = load_mni152_brain_mask(), smoothing_fwhm=8.0)
  model.fit(niftiImages, design_matrix=design_matrix)
  z_map_out = model.compute_contrast(contrast,output_type='z_score')
  # ReportObject = model.generate_report(contrast,f"{contrast} Report",
  #                       threshold=2.6,
  #                       alpha=0.001,
  #                       cluster_threshold=10,plot_type='glass')
  # ReportObject.save_as_html(save_name)

  os.makedirs(os.path.dirname(save_name), exist_ok=True)
  nib.save(z_map_out,save_name)

  return z_map_out


def run_GLM_single_visit(z_maps_path,USE_Z_MAPS,z_maps_save_path, secondLevelGLMDF, type_of_image):
  """
  Runs a GLM contrasting two groups (or movement) at a single visit, for all visits. Contrasts are saved to `z_maps_save_path` and first level z_maps 
  are gathered from `z_maps_path`.
  """

  if type_of_image == 'movement':
    contrast = 'constant'

    design_matrix, niftiImages = create_movement_design_matrix(z_maps_path, USE_Z_MAPS, secondLevelGLMDF)
    z_map_out = run_second_level_GLM(design_matrix,niftiImages,contrast,os.path.join(z_maps_save_path,type_of_image,f'movement_against_baseline'))
  else:
    for visit in ['visit1', 'visit4']:
      design_matrix, niftiImages = create_design_matrix_for_single_visit(z_maps_path, USE_Z_MAPS, secondLevelGLMDF,visit, type_of_image=type_of_image)
      for contrast in ['SBPr - chronic','SBPp - chronic','SBPr - SBPp', 'SBPr','SBPp','chronic']:
        z_map_out = run_second_level_GLM(design_matrix, niftiImages, contrast, os.path.join(z_maps_save_path,type_of_image,f'{visit}_{contrast.replace(" ", "")}'))


def run_GLM_across_visits(z_maps_path, USE_Z_MAPS, z_maps_save_path, secondLevelGLMDF, type_of_image):
  """
  Runs a GLM contrasting a single group between two visits. The inputs are the same as `run_GLM_single_visit`.
  """
  for group in ['SBPp', 'SBPr', 'chronic']:
    contrast = 'visit1' + '-' + 'visit4'
    design_matrix, niftiImages = create_design_matrix_across_visits(z_maps_path, USE_Z_MAPS, secondLevelGLMDF, first_visit='visit1', second_visit='visit4', group=group, type_of_image=type_of_image)
    z_map_out = run_second_level_GLM(design_matrix, niftiImages,contrast,os.path.join(z_maps_save_path,type_of_image,f'{group}_{contrast.replace(" ", "")}'))


def run_chronic_GLMs(z_maps_path,USE_Z_MAPS, z_maps_save_path, secondLevelGLMDF, type_of_image):
  """
  Runs a second level GLM contrasting all chronic patients against baseline. Saves the output to `z_maps_save_path/chronic_baseline`.
  """
  design_matrix, niftiImages = create_design_matrix_all_chronic(z_maps_path, USE_Z_MAPS, secondLevelGLMDF,type_of_image)
  z_map_out = run_second_level_GLM(design_matrix, niftiImages,'constant',
                                   os.path.join(z_maps_save_path,'chronic_baseline',f'{type_of_image}_baseline')
                                  )


def threshold_z_map(z_map, threshold, title, alpha, plot=True, cluster_threshold=10):
  """ Thresholds a z map, applying false positive correction and a cluster threshold. """
  z_map_out, _ = threshold_stats_img(z_map,alpha=alpha,threshold=threshold,
                                    height_control='fpr',cluster_threshold=cluster_threshold,two_sided=True)
  if False:
    plot_stat_map(z_map_out, threshold=threshold, title=title, cut_coords= cuts, draw_cross=False, display_mode='z')
    plt.show()
  return z_map_out


"""
# Cluster plotting Regions
"""

def get_resampled_atlas(z_map,atlas_img):
    """
    Uses nilearn's resample_to_img function to resample an atlas to the affine of a target `z_map`.
    """
    atlas_resamp = resample_to_img(atlas_img,target_img = z_map,interpolation='nearest')
    return atlas_resamp


def get_labelled_clusters(z_map,threshold, cluster_threshold, two_sided, atlas,data):
    # takes the resampled atlas and its data in

    cluster_table_df = get_clusters_table(z_map,threshold, cluster_threshold, two_sided)
    coords = cluster_table_df[['X','Y','Z']].to_numpy().astype(int)
    # print(coords)
    data_x,data_y,data_z = coord_transform(coords[:,0],coords[:,1],coords[:,2],np.linalg.inv(z_map.affine))
    data_coords = np.column_stack((data_x,data_y,data_z)).astype(int)
    # print(data_coords)
    values = data[data_coords[:,0], data_coords[:,1], data_coords[:,2]].astype(int)
    # print(values)
    # print(atlas['labels'])
    cluster_table_df['brain_regions'] = np.array(atlas['labels'])[values]

    return cluster_table_df


def plot_all_contrasts(z_maps_save_path, threshold, alpha,atlas, savePath, compute_regions: bool=True, type_of_image='error',plot=True,figsize=(10,5),save=True):

  full_cluster_table = pd.DataFrame(columns=['Cluster ID','X','Y','Z','Peak Stat','Cluster Size (mm3)','brain_regions','type_of_image','file'])
  ho_maxprob_atlas_img = image.load_img(atlas['maps'])
  z_map = nib.load(f'{z_maps_save_path}/error/visit4_SBPr-chronic.nii')
  atlas_resamp = get_resampled_atlas(z_map,ho_maxprob_atlas_img)
  data = atlas_resamp.get_fdata()

  #plotting all single_visit_contrasts
  for file in os.listdir(f'{z_maps_save_path}/{type_of_image}'):
    z_map = nib.load(f'{z_maps_save_path}/{type_of_image}/{file}')
    z_map_out = threshold_z_map(z_map,threshold, file, alpha,cluster_threshold=10)

    fig,ax = plt.subplots(1,figsize=figsize,dpi=500)
    Name = 'Difference signal' if type_of_image == 'error' else type_of_image
      
    #display = plot_glass_brain(z_map_out,colorbar=True, threshold=threshold,plot_abs=False,title=f'{Name} contrast, {file}, $\\alpha = 0.001$',figure=fig,axes=ax)
    try:
      contrast_1, contrast_3 = file[:-4].split('-')
      contrast_1, contrast_2 = contrast_1.split('_')
    except:
       contrast_1, contrast_2, contrast_3 = "","", ""
    contrast_2 = ('recovering' 
    if contrast_2 == "SBPr"
    else 'persisting' 
    if contrast_2 == "SBPp"
    else 'chronic')

    contrast_3 = ('recovering' 
    if contrast_3 == "SBPr"
    else 'persisting' 
    if contrast_3 == "SBPp"
    else 'chronic')


    display = plot_glass_brain(z_map_out,colorbar=True, threshold=threshold,plot_abs=False,title=f'{Name} contrast {contrast_1}, {contrast_2} vs {contrast_3} ',figure=fig,axes=ax)
    plt.tight_layout()
    if compute_regions:
      colours = ['green','orange','pink','yellow','purple','grey','turquoise','cyan','violet','brown','black','#AAB345','#CF4134','#BE43B2']
      labelled_cluster_table = get_labelled_clusters(z_map_out,threshold, cluster_threshold=10,two_sided=True,atlas=atlas,data=data)
      cluster_coords = labelled_cluster_table[['X','Y','Z']].to_numpy().astype(int)
      cluster_labels = labelled_cluster_table['brain_regions'].to_numpy().astype(str)
      # Add markers for each cluster region
      # assign clusters labels a colour
      unique_labels = list(set(cluster_labels))

      for coord, label in zip(cluster_coords, cluster_labels):
          display.add_markers([coord], marker_color=colours[unique_labels.index(label)], marker_size=10)

      legend_elements = [Line2D([0], [0], marker='o', color='w', label=unique_labels[i],markerfacecolor=colours[i], markersize=5) for i in range(len(unique_labels))]
      ax.legend(handles=legend_elements,fontsize=9,ncol = 3,loc='lower center', 
        bbox_to_anchor=(0.5, -0.1),fancybox=False,)
      if save:
        labelled_cluster_table.to_csv(f'{savePath}/{type_of_image}/{type_of_image}_{file}.csv')
      labelled_cluster_table['type_of_image'] = type_of_image
      labelled_cluster_table['file'] = file
      full_cluster_table = pd.concat([full_cluster_table,labelled_cluster_table],ignore_index=True)
      
    
    if save:
      fig.savefig(f'{savePath}/{type_of_image}/{type_of_image}_{file}.png',bbox_inches='tight')
    if plot:
      plt.show()
    else:
      plt.close()

  return full_cluster_table


def add_PAG_to_atlas(atlas_name = 'cort-maxprob-thr25-2mm'):
  atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
  mni_template = datasets.load_mni152_template(1)
  #template_file = mni_template['t1']

  PAG = nib.load("AAN_PAG_MNI152_1mm_v1p0_20150630.nii")

  ho_maxprob_atlas_img = image.load_img(atlas['maps'])
  PAG_resamp = image.resample_img(PAG,ho_maxprob_atlas_img.affine,interpolation='nearest',target_shape=ho_maxprob_atlas_img.shape)
  PAG_ROI = image.math_img("49 * img", img=PAG_resamp)

  combined_atlas = image.math_img("img1 + img2", img1=ho_maxprob_atlas_img, img2=PAG_ROI)
  atlas['maps'] = combined_atlas
  atlas['labels'] = atlas['labels'] + ['Periaqueductal Gray']
  return atlas


def plot_save_subcortical():
  atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
  plot=False
  savePath = 'SecondLevelContrastsSavedSubcortical'

  error_cluster_table = plot_all_contrasts(2.6,0.001, atlas, savePath, type_of_image='error',plot=plot,figsize=(8,3))
  pain_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath, type_of_image='pain',plot=plot,figsize=(8,3))
  #movement_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath, 'movement',plot=plot,figsize=(8,3))

  full_cluster_table = pd.concat([error_cluster_table,pain_cluster_table],ignore_index=True)
  cluster_list = list(set(full_cluster_table['brain_regions']))
  cluster_list.remove('Background')
  print(cluster_list)

def plot_save_cortical(PAG=False):
  if PAG:
    atlas = add_PAG_to_atlas('cort-maxprob-thr25-2mm')
  else:
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
  plot=False
  savePath = 'SecondLevelContrastsSavedCortical_FINAL'

  error_cluster_table = plot_all_contrasts(2.6,0.001, atlas, savePath, type_of_image = 'error',plot=plot,figsize=(8,3))
  pain_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath,type_of_image =  'pain',plot=plot,figsize=(8,3))
  movement_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath, type_of_image='movement',plot=plot,figsize=(8,3))

  full_cluster_table = pd.concat([error_cluster_table,pain_cluster_table],ignore_index=True)
  cluster_list = list(set(full_cluster_table['brain_regions']))
  cluster_list.remove('Background')
  print(cluster_list)


def plot_unlabelled_contrasts():
  atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
  plot=True
  savePath = 'SecondLevelContrastsSavedUnlabelled'
  label=False
  error_cluster_table = plot_all_contrasts(2.6,0.001, atlas, savePath,
    compute_regions=label, type_of_image = 'error',plot=plot,figsize=(8,3),save=False)
  pain_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath,
    compute_regions=label,type_of_image = 'pain',plot=plot,figsize=(8,3),save=False)
  movement_cluster_table = plot_all_contrasts(2.6,0.001, atlas,savePath,
    compute_regions=label,type_of_image = 'movement',plot=plot,figsize=(8,3),save=False)


def plot_chronic_against_baseline(z_maps_save_path, threshold=2.6,alpha=0.05, compute_regions=False,save=False,plot=True):
  atlas = add_PAG_to_atlas()

  z_map_error = nib.load(f'{z_maps_save_path}/chronic_baseline/error_baseline.nii')
  z_map_pain = nib.load(f'{z_maps_save_path}/chronic_baseline/pain_baseline.nii')
  
  ho_maxprob_atlas_img = image.load_img(atlas['maps'])
  atlas_resamp = get_resampled_atlas(z_map_error,ho_maxprob_atlas_img)
  data = atlas_resamp.get_fdata()

  #plotting all single_visit_contrasts
  types = ['error','pain']
  for i, z_map in enumerate([z_map_error, z_map_pain]):
    z_map_out,thresh = threshold_stats_img(z_map,alpha=alpha,height_control='fpr',cluster_threshold=10,two_sided=True)

    fig,ax = plt.subplots(1,figsize=(9,5),dpi=150)
    display = plot_glass_brain(z_map_out,colorbar=True, threshold=thresh,plot_abs=False,
                               title=f'{types[i]} against baseline, fpr: {alpha}',figure=fig,axes=ax)
    
    if compute_regions:
      colours = ['green','orange','pink','yellow','purple','grey','turquoise','cyan','violet','brown','#2031A6','#5554A2','orange','#111234','red','red','red']
      labelled_cluster_table = get_labelled_clusters(z_map_out,threshold, cluster_threshold=10,two_sided=True,atlas=atlas,data=data)
      labelled_cluster_table = labelled_cluster_table.replace('Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)','Juxtapositional Lobule Cortex')
      cluster_coords = labelled_cluster_table[['X','Y','Z']].to_numpy().astype(int)
      cluster_labels = labelled_cluster_table['brain_regions'].to_numpy().astype(str)
      # Add markers for each cluster region
      # assign clusters labels a colour
      unique_labels = list(set(cluster_labels))

      for coord, label in zip(cluster_coords, cluster_labels):
          display.add_markers([coord], marker_color=colours[unique_labels.index(label)], marker_size=10)

      legend_elements = [Line2D([0], [0], marker='o', color='w', label=unique_labels[i],markerfacecolor=colours[i], markersize=5) for i in range(len(unique_labels))]
      ax.legend(handles=legend_elements,fontsize=9,ncol = 3,loc='upper center', 
        bbox_to_anchor=(0.5, 0),fancybox=False,)
      # labelled_cluster_table.to_csv(f'{z_maps_save_path}/chronic_baseline/{types[i]}.csv')
      # labelled_cluster_table['type_of_image'] = types[i]
      # labelled_cluster_table['file'] = f'{z_maps_save_path}/chronic_baseline/{types[i]}.nii' 
      
    plt.tight_layout()
    if save:
      fig.savefig(f'{z_maps_save_path}/chronic_baseline/{types[i]}.png',bbox_inches='tight')
    if plot:
      plt.show()
    else:
      plt.close()
    