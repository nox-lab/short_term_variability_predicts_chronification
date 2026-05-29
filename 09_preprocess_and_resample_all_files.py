from load_confounds import Confounds
import os
import pandas as pd
import numpy as np
import nilearn
import nibabel as nb
import logging
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn import plotting

# 2023-02-03: changed this SIMEXP's load confounds has moved
from nilearn.interfaces.fmriprep import load_confounds

template = load_mni152_template()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_subject_sessions(subjects_to_not_consider = []):
    """
    Load subjects rejecting those that are in the list subjects_to_not_consider
    
    Parameters
    ----------
    subjects_to_not_consider : List
        e.g. ['099', '104', '121'], returned from just_check_output_folder()
    
    Returns
    -------
    subject_sessions : List
        e.g. ['output/sub-001/ses-visit1', 'output/sub-001/ses-visit2']
    """
    parent_dir = "output/"
    
    subject_sessions = []
    for subject_dir in sorted(os.listdir(parent_dir)):
        subject_dir_path = os.path.join(parent_dir, subject_dir)
        if os.path.isdir(subject_dir_path) and "sub" in subject_dir_path: # check for sub
            for visit_dir in sorted(os.listdir(subject_dir_path)):
                visit_dir_path = os.path.join(subject_dir_path, visit_dir)
                if os.path.isdir(visit_dir_path) and "ses" in visit_dir_path: # check for ses
                    if not any(subject_id in visit_dir_path for subject_id in subjects_to_not_consider):
                        subject_sessions.append(visit_dir_path)
    return subject_sessions #doesn't check whether all of the files are in there

def load_subject_files(subject_sessions):
    """
    Create dataframe for the subject and sessions for easy access of all files
    
    Parameters
    ----------
    subject_sessions : List
        e.g. ['output/sub-001/ses-visit1', 'output/sub-001/ses-visit2'], returned from load_subject_sessions()
        
    Returns
    -------
    df : pd.DataFrame()
    """
    df = pd.DataFrame()
    for session_dir in sorted(subject_sessions):
        dict_for_subject = {"subject": str(session_dir.split("/")[1][4:]), "session": str(session_dir.split("/")[2][-1])}
        try:
            for file in sorted(os.listdir(session_dir + "/func/")):
                if "confounds" in file:
                    complete_file_path = os.path.join(session_dir + "/func/", file)
                    if "mv" in file in file:
                        dict_for_subject["mv_confounds"] = complete_file_path
                    elif "sp_run-01" in file:
                        dict_for_subject["sp_run_01_confounds"] = complete_file_path
                    elif "sp_run-02" in file:
                        dict_for_subject["sp_run_02_confounds"] = complete_file_path
                    elif "sv" in file:
                        dict_for_subject["sv_confounds"] = complete_file_path
                    else:
                        pass
                    
                if "nii.gz" in file:
                    complete_file_path = os.path.join(session_dir + "/func/", file)
                    if "mv" in file and "brain_mask" in file:
                        dict_for_subject["mv_brain_mask"] = complete_file_path
                    elif "sp_run-01" in file and "brain_mask" in file:
                        dict_for_subject["sp_run_01_brain_mask"] = complete_file_path
                    elif "sp_run-02" in file and "brain_mask" in file:
                        dict_for_subject["sp_run_02_brain_mask"] = complete_file_path
                    elif "sv" in file and "brain_mask" in file:
                        dict_for_subject["sv_brain_mask"] = complete_file_path
                    elif "mv" in file and "desc-preproc_bold" in file:
                        dict_for_subject["mv_bold"] = complete_file_path
                    elif "sp_run-01" in file and "desc-preproc_bold" in file:
                        dict_for_subject["sp_run_01_bold"] = complete_file_path
                    elif "sp_run-02" in file and "desc-preproc_bold" in file:
                        dict_for_subject["sp_run_02_bold"] = complete_file_path
                    elif "sv" in file and "desc-preproc_bold" in file:
                        dict_for_subject["sv_bold"] = complete_file_path
                    else:
                        pass
        except Exception as e:
            print(e)
            continue
    
        new_row = pd.Series(dict_for_subject)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df


subject_sessions = load_subject_sessions()
subject_files_df = load_subject_files(subject_sessions)
subject_files_df


subject_files_df.isna().sum()

subject_files_df.loc[0]["mv_brain_mask"]

def check_nii_shapes(subject_files_df):
    stim_on_files = subject_files_df["sp_run_01_bold"].tolist()
    subjects_to_not_consider = []
    
    for file in stim_on_files:
        try:
            loaded_data = nb.load(file)
            if loaded_data.shape != (57, 68, 65, 244):
                print(file, loaded_data.shape)
                subjects_to_not_consider.append(file.split("/")[1].split("-")[1])
        except:
            logging.warning("didn't happen for", file)
            print("didn't happen for", file)
            
    return list(set(subjects_to_not_consider))

check_nii_shapes(subject_files_df)

def preprocess_file(subject_id = "097", task = "sp_run_01", session = "1"):
    #TODO Docstring for this
    bold_file = subject_files_df.loc[(subject_files_df.subject == subject_id) & (subject_files_df.session == session)][task + "_bold"].tolist()[0]
    mask_file = subject_files_df.loc[(subject_files_df.subject == subject_id) & (subject_files_df.session == session)][task + "_brain_mask"].tolist()[0]
    
    # got code from P9 here: https://github.com/SIMEXP/load_confounds/pull/51/commits/26467de1387ad29773ff394dcb62f40c03ebb088
    # added "high_pass" in strategy and n_motion doesn't exist
    confounds = load_confounds(bold_file,
                              strategy=["high_pass", "motion", "wm_csf", "global_signal"], 
                              motion="basic", 
                              # n_motion = 0,
                              wm_csf="basic",
                              global_signal="basic")
    # print("confounds shape:", confounds)
    # print(len(confounds), type(confounds[0]), confounds[0], type(confounds[1]), confounds[1], confounds[1].shape)
    # display(confounds[0])
    # print(confounds[1])
    t_r = 2.5
    high_pass= 0.006
    low_pass = None
    bold_img = nilearn.image.load_img(bold_file)
    bold_img = bold_img.slicer[:,:,:,4:]
    #plotting.plot_epi(bold_img.slicer[:, :, :, 50])
    confounds_matrix = confounds[0].iloc[4:]
    cleaned_bold = nilearn.image.clean_img(bold_img, 
                                              confounds=confounds_matrix, 
                                              detrend=False, 
                                              standardize=False, 
                                              low_pass=low_pass, 
                                              high_pass=high_pass,
                                              ensure_finite=True,
                                              mask_img=mask_file,
                                              t_r = t_r
                                             )
    cleaned_bold = nilearn.image.smooth_img(cleaned_bold, fwhm=7)
    logging.info("cleaned_bold shape:", cleaned_bold.shape)
    print("cleaned_bold shape:", cleaned_bold.shape)
    # plotting.plot_epi(cleaned_bold.slicer[:, :, :, 50])
    return cleaned_bold, mask_file


def preprocess_single_run(row, run, output_dir, index, subject_files_df):
    '''Function that preprocesses subject data based on their run and saves data in a column that states
    the run number
    input: row-> row in dataframe of all subjects
    run: either run01, run02
    output_dir -> directory for output path putting together subject, visit, run
    index: column index
    subject_files_df -> dataframe with output'''

    subject = "sub-" + row["subject"]
    visit = "ses-visit" + str(row["session"])
    output_path = os.path.join(output_dir, subject, visit, run, "cleaned_file.nii.gz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(output_path, "already exists!!!")
        subject_files_df.at[index,f"resampled_and_preprocessed_bold_{run}"] = output_path
    else:
        cleaned_bold, mask_file = preprocess_file(subject_id = row["subject"], task = f"{run}", session = row["session"])
        resampled = resample_to_img(cleaned_bold, template)
        resampled.to_filename(output_path)
        logging.info("resampled shape:", resampled.shape, "saved at:", output_path)
        print("resampled shape:", resampled.shape, "saved at:", output_path)
    subject_files_df.at[index, f'resampled_and_preprocessed_bold_{run}'] = output_path


def preprocess_all_files(subject_files_df):
    output_dir = "preprocessed_and_resampled_data/"
    os.makedirs(output_dir, exist_ok=True)

    for index, row in subject_files_df.iterrows():
        print("Preprocessing subject:", row["subject"], "visit:", row["session"])

        # Separate try blocks for each run
        try:
            print("preprocessing run 1")
            preprocess_single_run(row, "sp_run_01", output_dir, index, subject_files_df)
        except Exception as e:
            logging.warning("Run 01 failed for subject: %s, session: %s. Reason: %s", row["subject"], row["session"], str(e))
            print("Run 01 failed")

        try:
            print("preprocessing run 2")
            preprocess_single_run(row, "sp_run_02", output_dir, index, subject_files_df)
        except:
            logging.warning("DIDN'T HAPPEN FOR subject:", row["subject"], "session:", row["session"])
            print("DIDN'T HAPPEN FOR subject:", row["subject"], "session:", row["session"])

    return subject_files_df

subject_files_df_updated = preprocess_all_files(subject_files_df)


subject_files_df_updated.to_excel("subject_files_feb_2023.xlsx")