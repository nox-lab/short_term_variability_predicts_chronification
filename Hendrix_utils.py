import nibabel as nib
import os
import re
from nilearn import image, plotting
from matplotlib.colors import LinearSegmentedColormap
from nilearn.input_data import NiftiLabelsMasker
from nilearn.maskers import NiftiMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import surface, datasets
from nilearn.plotting import plot_surf_roi
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import matplotlib.image as mpimg
from sklearn.metrics import roc_auc_score
import seaborn as sns
import ast
import datetime
import json
from PIL import Image
from scipy.stats import t

from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


###################################Generating file paths ##############################################################

def obtain_base_paths(setup_style):

    # Define setup styles
    setup_style = "gaia"  # Change this based on your current machine

    # Define paths based on setup
    if setup_style == "hendrix_PC":
        base_path = 'F:/Coding projects/Nox Lab/sbp-main'
        tian_atlas_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T.nii.gz') 
        tian_label_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T_label.txt')
        z_map_path = os.path.join(base_path, 'data', 'Zscores')
        second_level_glm_path = os.path.join(base_path, 'data', 'secondLevelGLMDF.csv')
        run_mapping = os.path.join(base_path, 'Results', 'Results', 'Non_discretized_responses', 'Carl_preprocessed_responses', 'preprocessed_response_details.csv')
        rate_of_change_csv_path = os.path.join(base_path, 'data', 'Carl_preprocessed_responses', 'painrating_rate_of_change.csv')
        selected_labels_csv_path = os.path.join(base_path, 'data', 'selected_subjects_labels.csv')

    elif setup_style == "hendrix_mac":
        base_path = '/Users/hendrixwylde/Downloads/sbp-main-2'
        tian_atlas_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T.nii.gz')
        tian_label_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T_label.txt')
        z_map_path = os.path.join(base_path, 'data', 'Zscores')
        second_level_glm_path = os.path.join(base_path, 'data', 'secondLevelGLMDF.csv')
        run_mapping = os.path.join(base_path, 'Results', 'Results', 'Non_discretized_responses', 'Carl_preprocessed_responses', 'preprocessed_response_details.csv')
        rate_of_change_csv_path = os.path.join(base_path, 'data', 'Carl_preprocessed_responses', 'painrating_rate_of_change.csv')
        selected_labels_csv_path = os.path.join(base_path, 'data', 'selected_subjects_labels.csv')

    #TODO For Carl: we should figure out one place where we all keep secondlevelGLMdf -- we had decided to move it to intermediate files?
    # I have a version there

    elif setup_style == "carl":
        base_path = "/scratch/ca541/sbp"
        tian_atlas_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T.nii.gz')
        tian_label_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T_label.txt')
        z_map_path = os.path.join(base_path, 'Zscores')
        run_mapping = os.path.join(base_path, 'Results', 'Results', 'Non_discretized_responses', 'Carl_preprocessed_responses', 'preprocessed_response_details.csv')
        second_level_glm_path = os.path.join(base_path, 'data', 'secondLevelGLMDF.csv')

    elif setup_style == "gaia":
        base_path = "/scratch/gp565/sbp"
        tian_atlas_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T.nii.gz')
        tian_label_path = os.path.join(base_path, 'melbourne atlas', 'Tian_Subcortex_S4_3T_label.txt')
        z_map_path = os.path.join(base_path, 'Results', 'Carl_first_level_z_maps_resampled_mask_2')
        run_mapping = os.path.join(base_path, 'Results', 'Results', 'Non_discretized_responses', 'Carl_preprocessed_responses', 'preprocessed_response_details.csv')
        second_level_glm_path = os.path.join(base_path, 'Results', 'Intermediate_Files', 'secondlevelGLMdf.csv')

    else:
        # Default paths / file names
        tian_atlas_path = "Tian_Subcortex_S4_3T.nii.gz"
        tian_label_path = "Tian_Subcortex_S4_3T_label.txt"
        z_map_path = 'carl_z_maps/'
        run_mapping = os.path.join(base_path, 'Results', 'Results', 'Non_discretized_responses', 'Carl_preprocessed_responses', 'preprocessed_response_details.csv')
        second_level_glm_path = "secondLevelGLMDF.csv"

    return base_path, tian_atlas_path, tian_label_path, z_map_path, second_level_glm_path, run_mapping


def generate_file_paths(subject_id, visit_range, zmap_path, run_map):
    #! THIS WAS CHANGED FROM FIXED PATH TO BASE PATH, SO IF THE FUNCTION DOES NOT RUN THE 
    # PROBLEM IS LIKELY THAT YOU HAVE TO GIVE IT BASE_PATH

    """
    Generates file paths for the given subject across a specified range of visits.
    """
    paths = []
    for visit in visit_range:  # Use the dynamically provided visit range
        visit_str = f"visit{visit}"
        # Find matching run for this visit
        matching = [run for v, run in run_map if v == visit_str]
        if not matching:
            continue

        run = matching[0]  
        visit_path = os.path.join(zmap_path, f'sub-{subject_id:03d}', f'ses-visit{visit}', run, 'z_map.nii.gz')
        paths.append(visit_path)
        
    return paths


def generate_output_classifier_dir(base_path, classifier_name, distribution, data_type):
    '''   
    Function that creates an output directory for the results of the 
    classifier of choice if it does not exist, else it continues 

    Params:
    base_path: path to results set in the function obtain_base_paths()
    classifier_name: name of the classifier of choice
    distribution: either Real or Null distribution
    data_type: type of data used for the classification - either Ratings or fMRI

    returns:
    generated path where the results are stored
    '''
    #COMMENT: I make a folder with {distribution}_data and not directly store the output there because we might want 
    #to test the performance of the classifier given different parameters e.g. n_splits, and in this way they will
    #not be overwritten, and will be easily comparable.


    path_to_results = os.path.join(base_path, "Results", f"{data_type}_Classifiers_Output", f"{classifier_name}",
                                        f"{distribution}_data")
    os.makedirs(path_to_results, exist_ok=True)

    return path_to_results


def get_path_to_classifier_output(data, classifier_name, distribution):
    '''   
    Function that gets the path to the output of a classifier on either the real data or the null
    distribution - for either rating data or fMRI data

    params:
    classifier_name: name of the classifier of choice
    distribution: either Real or Null distribution
    data_type: type of data used for the classification - either Ratings or fMRI

    returns: the path to the directory where the results are stored

    '''
    output_dir = f"Results/{data}_Classifiers_Output/{classifier_name}/{distribution}_data/"

    return output_dir


def generate_output_significant_parcels(base_path, classifier_name, data_type):
    '''
    Function that generates the dir Intermediate_Files if it does not exist - and just continues if it does
    It generates a path for the csv file containing the significant parcels after permutation testing.

    Params:
    base_path: base path generated via setting "user" at the beginning of each file starting with "Hendrix" or "Gaia"
    classifier_name: name of the classifier that generated the classification, as it appears on the directory name
    data_type: Type of data classified, either "Ratings" or "fMRI"

    Returns:
    Path to a directory that stores the csv output of the permutation test.

    '''
    path_to_results = os.path.join(base_path, "Results", "Intermediate_Files", f"{data_type}_Classifiers_Significant_Output", f"{classifier_name}")
    os.makedirs(path_to_results, exist_ok=True)
    return path_to_results

###################################Load, aggregate subjects data. Get subject ID#####################################################################


def load_and_aggregate_subjects(subject_id, visit_range, run_mapping_path, base_path):
    #Open rating files as it contains the mapping to run number per visit 
    run_mapping_df = pd.read_csv(run_mapping_path, delimiter = '\t')

    #Here we pick run_02 in case a visit has both run1 and run2, choose keep = 'first' to pick run_01 instead
    run_mapping_df = run_mapping_df.drop_duplicates(subset = ['subject', 'visit'], keep = 'last')

    #Select the run of choice (run_mapping_df) contains either run_02 or run_01 when available for a visit
    subject_label = f"sub-{subject_id:03d}"
    visits_col = run_mapping_df[run_mapping_df["subject"] == subject_label]["visit"]
    runs_col = run_mapping_df[run_mapping_df["subject"] == subject_label]["run"]
    runs = list(zip(visits_col.tolist(), runs_col.tolist()))

    file_paths = generate_file_paths(subject_id, visit_range, base_path, runs)
    masker = NiftiMasker()
    visit_data = []
    
    for path in file_paths:
        if os.path.exists(path):
            nifti_img = nib.load(path)
            data = masker.fit_transform(nifti_img)
            visit_data.append(data)
        else:
            return None  # Return None if any visit is missing
    
    print(subject_id, len(visit_data))

    # Ensure all data has the same shape before aggregation
    if len(visit_data)>=2 and len(visit_data) <=4:  # Ensure there are 4 visits
        # Find the minimum shape across all visits
        min_shape = min([data.shape[1] for data in visit_data])
        # Trim each visit data to the minimum shape
        visit_data = [data[:, :min_shape] for data in visit_data]
        
        # Aggregate data across visits (mean across visits)
        aggregated_data = np.mean(visit_data, axis=0)

        return aggregated_data
    else:
        return None
    

def get_subject_ids(base_path, visit_range):
    '''  
    Function that extracts the subject id from folder name in every visit
    '''
    subject_ids = []
    subject_folders = [f for f in os.listdir(base_path) if f.startswith('sub-')]

    for folder in subject_folders:
        subject_id = int(folder.split('-')[1])  # Extract subject ID from folder name
        valid = True

        # Check if the subject has data for all visits in the visit range
        for visit in visit_range:
            visit_path = os.path.join(base_path, folder, f'ses-visit{visit}', 'z_map.nii.gz')
            if not os.path.exists(visit_path):
                valid = False
                break

        if valid:
            subject_ids.append(subject_id)

    return subject_ids


def get_labels(sub_ids: list, df):
    # returns the group label of all the subjects which you enter
    labels = []
    for sub, row in df.iterrows():
        if sub in sub_ids:
            if row['chronic'] == True:
                labels.append('-1')
            if row['SBPr'] == True:
                   labels.append(0)
            if row['SBPp'] == True:
                labels.append(1)
    return labels


def generate_labels(subjects, S_all, max_visits=5):
    labels = []
    for subject in subjects:
        label = S_all[subject]
        labels.extend([label] * max_visits)     
    return np.array(labels)


def ensure_string_keys(dictionary):
    """Ensure that all keys in a dictionary are strings."""
    return {key.decode() if isinstance(key, bytes) else key: value for key, value in dictionary.items()}


def select_subject_id_and_group_label(z_map_path, second_level_glm_path, run_map):
    #COMMENT: made this function to get some order, as things were called randomly in the code before
    #confused on why we had code that takes subject id as input and returns subject id
    #so just returning group labels, but left the chunk to get ids in.

    '''  
    Function that given an range of subjects (arbitrary, selected by ID) retrieves thier id and 
    group label  

    params:
    z_map_path : path to the z-map per subject
    second_level_glm_path: path to file input to second level glm
    range_selected: range of selected patients (arbitrary) default is total number of patients

    returns:
    List of subjects with their group (1 - 0)
    '''

    # Map each subject to its visit and run's fMRI data

    selected_subjects = []
    selected_labels = []

    #Get dataset with subjects' groups
    SecondLevelGLMDF = pd.read_csv(second_level_glm_path)

    #Filter selected subjects by ID range (arbitrary)
    for subject_id in range(1, len(SecondLevelGLMDF)+1):

        #get zmaps per subject and append the ids of these subjects to a list if the data exists
        #COMMENT for Carl: this will break for you, the paths are not correct I think
        data = load_and_aggregate_subjects(subject_id, visit_range=range(1,5), run_mapping_path = run_map, base_path = z_map_path)

        if data is not None:
            selected_subjects.append(subject_id)

    S_all = get_labels(selected_subjects, SecondLevelGLMDF)

    return S_all



#############Get Tian and Schaefer parcellations, combine them. Get combined parcellation for subject ################

def get_tian_parcellation(tian_atlas_path, tian_label_path):
    '''
    Function that fetches Schaefer atlas and returns images and labels
    of parcels
    '''
    # Load Tian atlas
    tian_img = nib.load(tian_atlas_path)
    
    # Load Tian labels
    with open(tian_label_path, 'r') as f:
        tian_labels = [line.strip() for line in f.readlines()]
    
    return tian_img, tian_labels


def get_schaefer_parcellation(n_rois=200):
    '''
    Function that loads the Tian Atlas and returns images and labels
    of parcels
    '''
    # Fetch Schaefer atlas
    schaefer_parc = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2, verbose=False)
    schaefer_img = nib.load(schaefer_parc['maps'])
    return schaefer_img, schaefer_parc['labels'].tolist()


def combine_parcellations(schaefer_img, tian_img, schaefer_labels, tian_labels):
    ''' 
    Function that combines the Schaefer and Tian parcellations and returns 
    combined images and labels
    '''
    # Resample Tian atlas to match the Schaefer resolution 
    tian_resampled = image.resample_to_img(tian_img, schaefer_img, interpolation='nearest')
    
    # Get data arrays
    schaefer_data = schaefer_img.get_fdata()
    tian_data = tian_resampled.get_fdata()
    
    # Increment Tian atlas labels to avoid overlap
    tian_data[tian_data > 0] += schaefer_data.max()
    
    # Combine the maps into a single NIfTI image
    combined_data = np.copy(schaefer_data)
    combined_data[tian_data > 0] = tian_data[tian_data > 0]
    combined_img = nib.Nifti1Image(combined_data, schaefer_img.affine)

    # Combine the labels
    combined_labels = schaefer_labels + tian_labels
    
    return combined_img, combined_labels


def get_combined_parcellation_for_X(X_list, tian_atlas_path, tian_label_path, n_rois=200, early_prediction=False,
                                    visit_2 = False, visit_3 = False):
    '''   
    Function that combines parcellations for a given subject
    '''
    schaefer_img, schaefer_labels = get_schaefer_parcellation(n_rois)
    tian_img, tian_labels = get_tian_parcellation(tian_atlas_path, tian_label_path)
    combined_img, combined_labels = combine_parcellations(schaefer_img, tian_img, schaefer_labels, tian_labels)
    nlb = NiftiLabelsMasker(combined_img, strategy="mean")

    if early_prediction:
        # Use only the first visit (visit1)
        R_combined_list = nlb.fit_transform(X_list[0])
        R_combined = np.mean(R_combined_list, axis=0)  

    if visit_2:
        # Use only the second visit (visit2)
        R_combined_list = nlb.fit_transform(X_list[1])
        R_combined = np.mean(R_combined_list, axis=0)  

    if visit_3:
        # Use only the third visit (visit3)
        R_combined_list = nlb.fit_transform(X_list[2])
        R_combined = np.mean(R_combined_list, axis=0)  

    else:
        # Average across all visits
        R_combined_list = [nlb.fit_transform(img) for img in X_list]
        R_combined = np.mean(R_combined_list, axis=0)

    # Define the masker and extract time series
    return R_combined, combined_labels

#############################Visualise Regions of Interest (ROIs) and Confusion Matrices####################################################################

def visualize_top_rois(avg_parcel_accuracies, std_parcel_accuracies, combined_labels, combined_img, top_n=20):
    # Sort the ROIs by their accuracies and select the top N
    sorted_rois = sorted(avg_parcel_accuracies.items(), key=lambda item: item[1], reverse=True)
    top_rois = sorted_rois[:top_n]
    
    top_labels = [roi[0] for roi in top_rois]
    top_accuracies = [roi[1] for roi in top_rois]
    top_std = [std_parcel_accuracies[label] for label in top_labels]
    
    # Plot the violin plots for each of the top N ROIs
    plt.figure(figsize=(15, 8))
    
    for i, label in enumerate(top_labels):
        # Assuming avg_parcel_accuracies[label] contains the list of ROC AUC scores for each iteration
        parcel_data = avg_parcel_accuracies[label]  # List of ROC AUC scores for the current parcel
        
        # Ensure parcel_data is a 1D array (list of scores)
        parcel_data = np.array(parcel_data).flatten()  # Flatten if it's a 2D array
        
        # Create a subplot for each parcel
        plt.subplot(top_n, 1, i + 1)  # Adjust the number of rows according to `top_n`
        
        # Create a violin plot for each parcel, showing distribution of ROC AUC scores
        sns.violinplot(data=parcel_data, color='lightblue', inner='quart')
        
        # Add lines for min and max values
        min_value = np.min(parcel_data)
        max_value = np.max(parcel_data)
        plt.axhline(min_value, color='green', linestyle='--', label='Min Value')
        plt.axhline(max_value, color='red', linestyle='--', label='Max Value')

        # Set title and labels
        plt.title(f"Violin Plot for Parcel: {label}")
        plt.xlabel('ROC AUC Score')
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the top N ROIs on a glass brain
    combined_data = combined_img.get_fdata()
    mask_data = np.zeros_like(combined_data)
    
    # Map labels to indices
    label_to_index = {label: i + 1 for i, label in enumerate(combined_labels)}
    
    # Apply mask for top_labels and assign unique values
    for label in top_labels:
        if label in label_to_index:
            idx = label_to_index[label]
            mask_data[combined_data == idx] = idx
    
    # Create a colormap with N unique colors
    colors = plt.cm.get_cmap('tab20', top_n)
    cmap = ListedColormap(colors(range(top_n)))
    
    masked_img = nib.Nifti1Image(mask_data, combined_img.affine)
    
    # Plot the combined atlas with only the top N ROIs
    plotting.plot_roi(masked_img, title=f'Top {top_n} ROIs in Combined Atlas', display_mode='ortho', draw_cross=False, cmap=cmap)
    plotting.show()

    # Plot the top 5 ROIs on a glass brain (using only top 5 ROIs for this visualization)
    top_5_labels = top_labels[:5]  # Select only the top 5 labels for glass brain
    mask_data = np.zeros_like(combined_data)

    for label in top_5_labels:
        if label in label_to_index:
            idx = label_to_index[label]
            mask_data[combined_data == idx] = idx
    
    masked_img = nib.Nifti1Image(mask_data, combined_img.affine)
    
    # Plot the top 5 ROIs on the glass brain, centered at x=0
    plotting.plot_glass_brain(masked_img, title=f'Top 5 ROIs on Glass Brain (x=0)', display_mode='lyrz', cut_coords=(0, 0, 0), colorbar=True, cmap=cmap)
    plotting.show()


def save_coordinates_as_nifti(label_to_index, brain_region, combined_data, combined_img, output_dir, roi_name):

    roi_index = label_to_index.get(brain_region, None)

    if roi_index is not None:
        binary_mask_data = (combined_data == roi_index).astype(np.uint8)
        binary_mask_img = nib.Nifti1Image(binary_mask_data, combined_img.affine)

        out_path = os.path.join(output_dir, f"{roi_name}_mask.nii.gz")

        nib.save(binary_mask_img, out_path)
        
        print(f"Saved: {out_path}")
    else:
        print(f"[WARNING] ROI name '{roi_name}' not found in label_to_index.")


def save_atlas_network_name(roi_name, top_rois_df, duplicated_parcels):
    """     
    Function that saves the schaefer and tian parcel names to a network name.
    This will later be turned into roi-specific definitions in the function
    "network_to_roi_name".
    """
    if "/" in str(roi_name):
        roi_name = roi_name.split("/")
        first_name_bit = str(roi_name[0])
        second_name_bit = str(roi_name[1])
        roi_name = first_name_bit + second_name_bit
        roi_name = roi_name.split(" ")
        roi_name = roi_name[0] + " " + roi_name[1] + " " + roi_name[3] + " " + roi_name[4] + " "+ roi_name[5]

    #Ensure that even duplicate parcels are stored separately if they belong to the same roi
    if roi_name in duplicated_parcels:

        original_roi_name = roi_name

        roi_number= top_rois_df[top_rois_df["region_name"] == roi_name]["brain_region"].iloc[0].split("_")[-1]
        brain_region_name = top_rois_df[top_rois_df["region_name"] == roi_name]["brain_region"].iloc[0] 

        roi_name = f"{roi_name}" + "_" + f"{roi_number}" 

        #Exception for specific case of identical names but ventral vs distal 
        if brain_region_name == "7Networks_RH_Cont_PFCv_1":
            roi_name = f"{original_roi_name}" + "_" + "ventral" + f"{roi_number}" 
        if brain_region_name == "7Networks_RH_Cont_PFCl_1":
            roi_name = f"{original_roi_name}" + "_" + "lateral" + f"{roi_number}" 
    print(roi_name)

    return roi_name


def visualize_top_rois_from_performance_df(top_rois_df, base_path, combined_labels, combined_img, top_n=20,
                                           model = "GBM",
                                           save_nifti = False,
                                           plot_glass_brain = False,
                                           do_surface_plot = False):
    """
    Visualizes the top N ROIs using violin plots (individual_scores) and overlays them on the atlas.
    Works with a DataFrame containing brain_region, performance_stats with individual_scores (nested lists).
    """
    if "region_name" not in top_rois_df.columns:
        top_rois_df["region_name"] = top_rois_df["brain_region"]

    #Output directory for mz3 files, which will be turned into surface plots using MRIcroGL
    os.makedirs(os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{model}"), exist_ok=True)
    output_dir = os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{model}")

    # Extract top N rows only
    top_rois_df = top_rois_df.sort_values(by="mean_performance", ascending=False)[:top_n]
    top_labels = top_rois_df["brain_region"].tolist()

    # Compute mean ROC AUC per iteration for each parcel
    parcel_means = {}
    
    combined_labels = [label.decode() if isinstance(label, bytes) else label for label in combined_labels]
    label_to_index = {label: i + 1 for i, label in enumerate(combined_labels)}
    combined_data = combined_img.get_fdata()

    #Identify duplicate brain regions - multiple parcels from same roi
    duplicate_mask = top_rois_df["region_name"].duplicated(keep=False)
    duplicated_parcels = top_rois_df[duplicate_mask]["region_name"].tolist()

    for idx, row in top_rois_df.iterrows():
        brain_region = row["brain_region"]
        roi_name = row["region_name"]
        stats = row["performance_stats"]

        roi_name = save_atlas_network_name(roi_name, top_rois_df, duplicated_parcels)

        top_rois_df.at[idx, "region_name"] = roi_name

        if save_nifti:
            save_coordinates_as_nifti(label_to_index, brain_region, combined_data, combined_img, output_dir, roi_name)

        #Checking that the performance_stats is actually a dict, else making it one
        if type(stats) != dict:
            stats = ast.literal_eval(stats)

        individual_scores = stats["individual_scores"]

        # Average across folds for each iteration
        mean_per_iteration = [np.mean(fold_scores) for fold_scores in individual_scores]

        parcel_means[brain_region] = mean_per_iteration  

        #Plot a scatterplot in case there is only one mean value across folds (performed one iteration of 5 folds)
        if len(mean_per_iteration) == 1:
            plot_type = "scatterplot"

        #Plot a violin plot in case there are multiple mean values across folds (performed multiple iterations of 5 folds each)
        else:
            plot_type = "violinplot"


    df_top_15 = pd.DataFrame(parcel_means)

    if plot_type == "scatterplot":
        # Reshape for plotting
        df_melted = df_top_15.melt(var_name="Brain Region", value_name="ROC AUC")

        plt.figure(figsize=(18, 6))
        ax = sns.stripplot(
            data=df_melted,
            x="Brain Region",
            y="ROC AUC",
            jitter=True,
            size=10,
            color="crimson"
        )
        plt.xticks(rotation=90)
        plt.ylim(0, 1.0)
        plt.title(f"Mean ROC AUC across folds for Top {top_n} ROIs")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    elif plot_type == "violinplot":
        
        # Violin plot
        plt.figure(figsize=(16, 8))
        sns.violinplot(data=df_top_15, inner='quart', palette='pastel', cut=0)
        sns.stripplot(data=df_top_15, color='black', size=4, jitter=True)
        plt.xticks(rotation=90)
        plt.ylim(0, 1.0) 
        plt.title(f"Violin Plots for Top {top_n} ROIs")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    # --- Atlas Plotting ---
    mask_data = np.zeros_like(combined_data)


    # Build mask for top labels
    for label in top_labels:
        if label in label_to_index:
            idx = label_to_index[label]
            mask_data[combined_data == idx] = idx
   
    # Color map
    colors = plt.cm.get_cmap('tab20', top_n)
    cmap = ListedColormap(colors(range(top_n+1)))

    masked_img = nib.Nifti1Image(mask_data, combined_img.affine)

    # Plot ROI overlay
    plotting.plot_roi(masked_img, title=f'Top {top_n} ROIs in Combined Atlas',
                      display_mode='ortho', draw_cross=False, cmap=cmap)
    plotting.show()

    # --- Glass Brain Plot (Top 5 Only) ---
    top_5_labels = top_labels[:5]
    mask_data = np.zeros_like(combined_data)

    for label in top_5_labels:
        if label in label_to_index:
            idx = label_to_index[label]
            mask_data[combined_data == idx] = idx

        else:
            print(f"Warning: Label '{label}' not found in atlas labels!")

    masked_img = nib.Nifti1Image(mask_data, combined_img.affine)
    

    if plot_glass_brain:
        plotting.plot_glass_brain(
            masked_img,
            title='Top 5 ROIs on Glass Brain (x=0)',
            display_mode='lyrz',
            cut_coords=(20, 20, 20),  # You can tweak these values
            colorbar=True,
            cmap=cmap
        )
        plotting.show()

    return top_rois_df


def display_top_ROIs_from_classifier(classifier_performance, n, setup = False):
    '''  
    Function that returns the adjusted df containing classifier output
    and top n parcels

    Params:
    classifier_performance: df returned by main() summarising classifier performance over iterations
    and folds
    n: number of top parcels by mean performance that are plotted

    Returns:
    setup_info: Information concerning the setup of the classifier and data that produced the df
    classifier_df: Adjusted df containing performance of the classifier on each parcel
    top_parcels: top n parcels extracted from classifier_df based on their mean_performance
    '''
    #Load df
    classifier_performance.reset_index(inplace=True)
    classifier_df = classifier_performance.rename(columns={"index":"brain_region"})
    classifier_df = classifier_df.iloc[4:,:]

    if setup:
        # separate setup info with classifier type specified from rest of the df
        setup_info = classifier_performance.iloc[:4,:1]
        classifier_df = classifier_df.drop(["setup_info"], axis=1)
        classifier_df["mean_performance"] = classifier_df["performance_stats"].apply(
            lambda x: x["mean"] if isinstance(x, dict) else ast.literal_eval(x)["mean"]
            )
    else:
        setup_info = "Not available"
        classifier_df["mean_performance"] = classifier_df["individual_scores"].apply(lambda x: x["mean"])

    #Extract the top 20 ROIs based on mean performance 
    classifier_df_sorted = classifier_df.sort_values(by="mean_performance", ascending=False)
    top_parcels = classifier_df_sorted[:n]

    return setup_info, classifier_df, top_parcels
    

def barplot_top_ROIs(df, nr_top_parcels, model_used = "GBM"):
    '''  
    Function that plots a barplots featuring the name of the ROI and the mean ROC AUC value

    Params:
    df: dataframe containing the name of the brain region and mean ROC AUC performance over iterations
    nr_top_parcels: number of brain regions that want to be shown
    '''
    if model_used == "GBM" or model_used =="GBM_early_prediction":
        colour = "#B955B4"
    else:
        colour = "#E3C61F"
    
    # --- Plot top ROIs
    top_df = df.sort_values(by="mean_performance", ascending=False)[:nr_top_parcels]
    top_parcels_plot = top_df[:nr_top_parcels]

    # Plot top ROIs
    plt.figure(figsize=(12, 6))
    plt.barh(top_parcels_plot ["brain_region"], top_parcels_plot ["mean_performance"], color=colour)
    plt.xlabel("Mean Accuracy (ROC AUC Score)")
    plt.title("Top ROIs by Accuracy")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.show()


def plot_human_labels(significant_df, base_path, mapping, model="GBM", prediction="all_visits", FDR_method="BH", error_bar = True):
    # Parse performance_stats and compute standard deviation over folds
    std_list = []

    if model == "GBM" or model == "early_GBM":
        #"#B955B4"
        colour_model ="#58B955"
        
    if model == "visit2_GBM":
        colour_model = "#E78E45"
    
    if model == "early_pred_GBM" and prediction == "with_chronics":
        colour_model ="#55AFB9"

    if model == "visit2_GBM" and prediction =="with_chronics":
        colour_model ="#8755B9"

    else:
        colour_model = "#E3C61F"

    #Extract the index of the schaefer and tian network region saves in the best parcels dataframe

    if prediction!="with chronics":
        significant_df["roi_anatomical_name"] = significant_df["region_name"].map(mapping)
    
    if prediction == "with_chronics":
        significant_df["roi_anatomical_name"] = significant_df["brain_region"].map(mapping)

    significant_df["plot_label"] = (
        significant_df.groupby("roi_anatomical_name").cumcount().astype(str) + " - " + significant_df["roi_anatomical_name"] 
        )
    # sorted_df = sorted_df.dropna(subset=["plot_label"])
    
    if error_bar:
        for idx, row in significant_df.iterrows():
            stats_dict = ast.literal_eval(row["performance_stats"])
            scores = stats_dict["individual_scores"][0]
            std_list.append(np.std(scores, ddof=1))
            
            #where 5 is the number of k-folds
            st_error = [x / np.sqrt(5) for x in std_list]
            t_val = t.ppf(1 - 0.025, df=5 - 1)
        
        # Add stds to DataFrame
        significant_df = significant_df.copy()
        significant_df["standard_error"] = st_error
        significant_df["std_over_folds"] = std_list
        significant_df["CI_lower"] = significant_df["mean_performance"] - t_val * significant_df["standard_error"]
        significant_df["CI_upper"] = significant_df["mean_performance"]  + t_val * significant_df["standard_error"]

        significant_df["CI"] = list(zip(
        significant_df["mean_performance"] - t_val * significant_df["standard_error"],
        significant_df["mean_performance"] + t_val * significant_df["standard_error"]
        ))

        # Sort by mean ROC AUC descending
        sorted_df = significant_df.sort_values("mean_over_folds", ascending=False)
        print(sorted_df["plot_label"])

        #xerr = np.array([
        # sorted_df["mean_over_folds"] - sorted_df["CI_lower"],
        # sorted_df["CI_upper"] - sorted_df["mean_over_folds"]
        # ])
    

        # Plot with error bars
        plt.figure(figsize=(12, 6))
        plt.barh(sorted_df["plot_label"], sorted_df["mean_over_folds"],xerr= significant_df["standard_error"],
                  color=colour_model, zorder=2, capsize = 5)

        plt.yticks(
        ticks=np.arange(len(sorted_df)),
        labels=sorted_df["roi_anatomical_name"]
        )
        
        plt.xlabel("Mean Accuracy (ROC AUC Score)")
        plt.xlim(0.5, 1)
        plt.gca().invert_yaxis()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "Figures", f"{model}_{FDR_method}_{prediction}_errbars.png"))
        plt.show()
            
    else:
        sorted_df = significant_df.sort_values("mean_over_folds", ascending=False)
    
        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(sorted_df["plot_label"], sorted_df["mean_over_folds"], color='skyblue')
        plt.yticks(
        ticks=np.arange(len(sorted_df)),
        labels=sorted_df["region_name"]
        )
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel("Mean Accuracy (ROC AUC Score)")
        plt.xlim(0.5, 1)
        plt.gca().invert_yaxis()  # This keeps the highest at the top after horizontal bar plot
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"{model}_{FDR_method}_{prediction}_significant.png"))
        plt.show()


def boxplot_ROC_AUC_performance(df, model):

    all_data = []

    if model == "GBM" or model == "early_GBM":
        #"#B955B4"
        colour_model ="#58B955"
    if model == "visit2_GBM":
        colour_model = "#E78E45"
    else:
        colour_model = "#EEF833"

    #Exclusively pick top 12
    df = df.copy().iloc[:12,:]

    for idx, roi in df.iterrows():

        perf_stats = roi["performance_stats"]
        roi_name = roi["roi_anatomical_name"]

        perf_stats = ast.literal_eval(perf_stats) 
        scores = perf_stats["individual_scores"][0]

        for score in scores:
            all_data.append({"ROI": roi_name, "AUC": score})

    #creating DataFrame for plotting top 12
    plot_df = pd.DataFrame(all_data)

    #plot boxplot with overlapping dots (perfomance in each fold)
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="ROI", y="AUC", data=plot_df, showfliers=False, color=colour_model)
    sns.stripplot(x="ROI", y="AUC", data=plot_df, color="black", alpha=0.6, jitter=True)

    plt.axhline(y=0.5, color='#d3d3d3', linestyle = "--")

    plt.ylabel("ROC AUC (5-folds)", fontsize=14)
    plt.xlabel(" ")
    plt.ylim(0.15, 1.02)
    plt.legend().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.show()

        

# Add surface_plot_path to the top parcels df to produce cm+surface plot figure
def get_surface_plot_path(top_parcels_df, base_path, model = "GBM", type_prediction="all_visits"):
    """  
    Add surface_plot_path to the top parcels df to produce cm+surface plot figure
    """

    #Intiialise column to store paths to surface plots
    top_parcels_df["surface_plot_path"] = " "

    list_rois = []

    #Path to where the plots are saved for each model and all_visits or early prediction
    path_directory = os.path.join(base_path, "Results", "Intermediate_Files",
                                   "Surface_Plots", f"{model}_{type_prediction}")
    
    if os.path.isdir(path_directory):
        files_paths = os.listdir(path_directory)

        for name in top_parcels_df["region_name"]:

            if "/" in str(name):

                name_first_bit = str(name).split("/")[0]
                name_second_bit = str(name).split("/")[1]
                name = name_first_bit + name_second_bit
                name = name.split(" ")
        
                name = name[0] + " " + name[1] + " " + name[3] + " " + name[4] + " " + name[5]
            
            #Loop through every file in the folder
            for roi_name in files_paths:

                #Adjusting name of file in folder to format name_surface_plot.png
                if "/" in str(roi_name):

                    roi_name_first_bit = str(roi_name).split("/")[0]
                    roi_name_second_bit = str(roi_name).split("/")[1]
                    roi_name = roi_name_first_bit + roi_name_second_bit 
                    roi_name = roi_name.split(" ")
            
                    roi_name = roi_name[0] + " " + roi_name[1] + " " + roi_name[3] + " "+roi_name[4] + " " + roi_name[5]
                
                #Matching ROI name in the dataframe to the surface plot file 
                if roi_name == f"{name}_surface_plot.png":
                    
                    list_rois.append(roi_name)


        top_parcels_df["surface_plot_path"] = list_rois

        return top_parcels_df
    else:
        return ("Directory with surface plots does not exist!")
        

#--- Confusion matrix for preliminary plotting and testing of classifiers

def show_confusion_matrix(cm, roi_name):
    '''
    Function that plots a confusion matrix of a classifier's output

    params:
    cm: computed confusion matrix 
    parcel_label: name of the parcel analysed

    returns: 
    plot of the confusion matrix of the classifier's performance
    '''
    cm = np.array(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Top Parcel: {roi_name}')


def confusion_matrix_per_fold(top_parcels):
    '''  
    Function that computes the sum of confusion matrices from different folds.
    If multiple iterations are performed, the confusion matrix is the sum of all the matrices in each fold and iteration.

    Params:
    top_parcels : top n parcels according to ROC AUC mean performance
    surf_plot_directory: path to the directory where the png of the surface plots are stored

    Returns: 
    Figure containing a confusion matrix representing the sum of all confusion matrices obtained in each
    iteration and fold. If surface_plot is enabled, it will produce a figure with surface plot next to 
    the corresponding ROI's confusion matrix.
    '''
    for idx, row in top_parcels.iterrows():
        roi = row["brain_region"]
        confusion_matrices = row["confusion_matrices"]

        # Convert each confusion matrix to a numpy array
        matrices_np = [np.array(cm) for cm in confusion_matrices if cm is not None]

        # Average element-wise over all 100 matrices
        sum_matrix = np.sum(matrices_np, axis=0)

        show_confusion_matrix(sum_matrix, roi_name=roi)


# --- Confusion matrix and surface plot figure
def show_cm(cm, ax, model="GBM", type_prediction = "visit2"):
    '''
    Function that plots a confusion matrix of a classifier's output
    to be plotted next to a surface plot

    params:
    cm: computed confusion matrix 
    model: model used for classification
    type_prediction: visit used for prediction

    returns: 
    plot of the confusion matrix of the classifier's performance
    '''
    cm = np.array(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    #lavender_colour = "#6006B5"

    if "GBM" in str(model) and "visit2" not in str(type_prediction):
        colour = LinearSegmentedColormap.from_list("lavender_violet", ["#E6E6FA", "#58B955"])

    elif "GBM" in str(model) and "visit2" in str(type_prediction):
        colour = LinearSegmentedColormap.from_list("lavender_violet", ["#E6E6FA", "#E78E45"])

    else:
        colour = LinearSegmentedColormap.from_list("lavender_violet", ["#E6E6FA", "#BFA307C2"])

    disp.plot(cmap=colour, ax=ax, colorbar=False)

    for text in disp.text_.ravel():
        text.set_color("black")
    # disp.plot(cmap=plt.cm.Purples, ax=ax, colorbar=False)


def show_surface_plot(surface_plot_path, ax):
    """
    Function that opens the png figure of the ROI overlaying
    a surface plot produced with MRIcroGL.
    Params:
        - surface_plot: path to the surface plot with the given ROI as 
          overlay 
    Returns: 
        One plot with surface plot 
    """
    surface_img = mpimg.imread(surface_plot_path)
    ax.imshow(surface_img)
    ax.axis('off')


def confusion_matrix_and_surface_plot_per_fold(top_parcels, base_path, model = "GBM", type_prediction = "all_visits"):
    '''  
    Function that computes the sum of confusion matrices from different folds.
    If multiple iterations are performed, the confusion matrix is the sum of all the matrices in each fold and iteration.

    Params:
    top_parcels : top n parcels according to ROC AUC mean performance
    surf_plot_directory: path to the directory where the png of the surface plots are stored

    Returns: 
    Figure containing a confusion matrix representing the sum of all confusion matrices obtained in each
    iteration and fold. If surface_plot is enabled, it will produce a figure with surface plot next to 
    the corresponding ROI's confusion matrix.
    '''
    for _, row in top_parcels.iterrows():
        confusion_m_row = row["confusion_matrices"]
        confusion_matrices = ast.literal_eval(confusion_m_row)
        
        roi_name = row["roi_anatomical_name"]
        path_to_surface_plot = row["surface_plot_path"]
        
        #Getting the path to the PNG with combined parcels 
        split_path = str(path_to_surface_plot).split("_")

        if len(split_path) == 3:
            path_to_surface_plot = str(split_path[0]) + "_" + "surface_plot.png"

        #Obtain path to PNGs of the surface plot, produced with MRIcroGL
        surf_plot_path = os.path.join(base_path, "Results", "Intermediate_Files",
                                   "Surface_Plots", f"{model}_{type_prediction}",
                                     path_to_surface_plot)

        if not confusion_matrices or not isinstance(confusion_matrices, list):
            print(f"No confusion matrices for {roi_name}")
            continue
        
        # Convert each confusion matrix to a numpy array
        matrices_np = [np.array(cm) for cm in confusion_matrices if cm is not None]

        # Average element-wise over all 100 matrices
        sum_matrix = np.sum(matrices_np, axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(3, 3), gridspec_kw={'width_ratios': [2, 0.8], 'wspace': 0.5})

        show_surface_plot(surf_plot_path, axes[0])
        show_cm(sum_matrix, ax=axes[1], model = model, type_prediction = type_prediction)
        fig.text(0.5,0.72, roi_name, ha='center', va='top', fontsize=11)
        #fig.save(f"Figures/Surface_and_CM/{model}_{type_prediction}/cropped_{roi_name}.png")
        #fig.suptitle(f"{roi_name}", fontsize=14, y=0.8)  # Moves title closer to plots
        plt.tight_layout() 
        plt.show()
        

def schaefer_to_brain_region(label):

    """
    Function that translates schaefer parcels into anatomical
    names
    """
    # Hemisphere mapping
    hemi_map = {
        "LH": "Left",
        "RH": "Right"
    }

    # Subregion (anatomical) mapping — expand this as needed
    region_map = {
        "Vis": "Visual Cortex",
        "SomMot": "Somatomotor Cortex",
        "DorsAttn": "Parietal Cortex",
        "SalVentAttn": "Salience / Ventral Attention Network",
        "Limbic": "Limbic System",
        "Cont": "Frontoparietal Control Network",
        "Default": "Default Mode Network",
    }

    # Parse label
    parts = label.split('_')
    if len(parts) < 4:
        return "Unknown Region"

    hemisphere= parts[1]
    region = parts[2]
    number = parts[-1]
    

    #Map to subregion of the brain names
    if region in region_map:
        region = region_map[region]

    #Map to hemisphere names
    if hemisphere in hemi_map.keys():
        hemisphere= hemi_map[hemisphere]

    if label =="7Networks_RH_Default_Par_3":
        return "Default Mode Network Parietal Region_3"
    
    return f"{hemisphere} {region}_{number}"



def tian_to_brain_region(label):

    if label == "mAMY-lh":
        return "Left Middle Amygdala"
    
    if label == "HIP-head-m2-lh":
        return "Left Hippocampal Head, Medial Cluster 2"

    hemi_map = {"lh": "Left", "rh": "Right"}
    parts = label.split("-")
    
    if len(parts) != 3:
        return "Unknown Region"
    
    structure_code, subregion_code, hemi_code = parts

    region_map = {
        "THA": "Thalamus",
        "PTH": "Posterior Thalamus",
        "STN": "Subthalamic Nucleus",
        "SN": "Substantia Nigra",
        "GP": "Globus Pallidus",
        "PUT": "Putamen",
        "CAU": "Caudate",
        "AMY": "Amygdala",
        "HIP": "Hippocampus",
        "BS": "Brainstem",
        "NAc": "Nucleus Accumbens"
    }
    

    subregion_map = {
        "VPl": "Ventral Posterolateral",
        "VP": "Ventroposterior",
        "VPm": "Ventral Posteromedial",
        "DAm": "Dorsal Anterior Medial",
        "DAl": "Dorsal Anterior Lateral",
        "VL": "Ventral Lateral",
        "MD": "Mediodorsal",
        "LGN": "Lateral Geniculate",
        "MGN": "Medial Geniculate",
        "CM": "Centromedia",
        "Pf": "Parafascicular",
        "VA": "Ventral Anterior",
        "VAs": "Ventral Anterior",
        "shell": "Shell",
        "head": "Head"
    }

    region = parts[0]
    subregion = parts[1]
    hemisphere = parts[2]

    if label == "mAMY-lh":
        subregion = "m"
        region = "AMY"
        hemisphere = "lh"

    if region in region_map:
        region = region_map[region]

    #Map to hemisphere names
    if hemisphere in hemi_map:
        hemisphere= hemi_map[hemisphere]

    if subregion in subregion_map:
        subregion = subregion_map[subregion]


    return f"{hemisphere} {subregion} {region}"


def parcel_to_network_name(df):
    region_names = []

    for label in df["brain_region"]:
        if "7Networks" in label:
            region_names.append(schaefer_to_brain_region(label))
        else:
            region_names.append(tian_to_brain_region(label))

    df["region_name"] = region_names
    
    return df



#Intersecting regions
def find_intersecting_regions(gbm_output, svc_output):
    """ 
    Function that finds the intersecting regions
    in two classifier outputs
    """
    intersecting_regions = []

    for region in gbm_output:

        if region in svc_output.tolist():
            intersecting_regions.append(region)

    for region in svc_output:
        if region in gbm_output.tolist():
            intersecting_regions.append(region)

    return set(intersecting_regions)


def display_intersecting_regions(base_path, gbm_output, svc_output, gbm_mapping, svc_mapping, visits = "all visits", save_name= "all_visits"):

    means = []
    stds = []
    colours=  []
    labels = []

    #violet = "#B955B4"
    colour_gbm = "#58B955"
    colour_svc = "#E3C61F"

    figure_network_names = {"Right Frontoparietal Control Network_3": "Frontoparietal Control Network",
                     "Left Visual Cortex": "Visual Processing Network",
                     "Right Default Mode Network": "Default Mode Network",
                     "Right Frontoparietal Control Network": "Frontoparietal Control Network",
                     "Right Somatomotor Cortex": "Somatomotor Network"}

    all_visits_intersection = list(find_intersecting_regions(gbm_output["region_name"], svc_output["region_name"]))
    
    for name in sorted(all_visits_intersection):

        if name in gbm_output["region_name"].tolist():
            gbm_row = gbm_output[gbm_output["region_name"]==name].iloc[0]
            gbm_performance = gbm_row["mean_performance"]
            gbm_std = ast.literal_eval(gbm_row["performance_stats"])["std"]
            gbm_figure_name = figure_network_names.get(gbm_row["region_name"], gbm_row["region_name"])

            means.append(gbm_performance)
            stds.append(gbm_std)
            colours.append(colour_gbm)
            labels.append(gbm_figure_name)


        if name in sorted(svc_output["region_name"].tolist()):
            svc_row = svc_output[svc_output["region_name"]==name].iloc[0]
            svc_performance = svc_row["mean_performance"]
            svc_std = ast.literal_eval(svc_row["performance_stats"])["std"]
            #svc_anatomical_name = svc_mapping.get(svc_row["region_name"], svc_row["region_name"])

            means.append(svc_performance)
            stds.append(svc_std)
            colours.append(colour_svc)
            labels.append(" ")
        
    # Plot
    y_pos = np.arange(len(labels))

    plt.figure(figsize=(10, 0.8 * len(labels)))  # Scales with number of parcels
    plt.barh(y_pos, means, xerr=stds, color=colours, edgecolor='black', capsize=4)
    
    plt.yticks(ticks=y_pos, labels=labels)
    plt.xlabel("Mean Accuracy (ROC AUC Score)")
    plt.xlim(0.5, 1)
    plt.gca().invert_yaxis()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.title(f"Functionally conn {visits}")
    plt.grid(False)
    # plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colour_gbm, edgecolor='black', label='GBM'),
        Patch(facecolor=colour_svc, edgecolor='black', label='SVC')
    ]
    plt.legend(handles=legend_elements, title="Classifier",loc='upper left',
    bbox_to_anchor=(1.01, 1),  # Pushes legend outside the plot
    borderaxespad=0.)
    plt.tight_layout(pad=1)

    # Save and show
    try:
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "Figures", f"GBM_vs_SVC_{save_name}_intersections.png"))
        plt.show()
    except:
          print("No intersecting networks")


def extract_coordinates_from_nifti(base_path, model="GBM", type_prediction=""):
    """
    Extract center-of-mass MNI coordinates from individual ROI NIfTI files.

    Parameters:
        base_path (str): Base directory path
        model (str): Model name (e.g., "GBM", "SVC")
        type_prediction (str): Optional suffix for other models

    Returns:
        List of tuples: (roi_name, MNI coordinate)
    """
    import os
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import center_of_mass
    from nilearn.image import coord_transform

    # Get path to ROI NIfTI files (GBM and SVC run on all visits is saved as GBM and SVC)
    if model in ["GBM", "SVC"] and type_prediction == " ":
        roi_dir = os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{model}")
    
    #GBM and SVC run on early prediction is saved as early_pred_GBM and early_pred_SVC so type prediction = early_pred
    if model in ["GBM", "SVC"] and type_prediction != " ":
        roi_dir = os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{type_prediction}_{model}")

    if model in ["GBM", "SVC"] and type_prediction != "early_pred_with_chronic":
        roi_dir = os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{type_prediction}_{model}")

    if model in ["GBM", "SVC"] and type_prediction == "visit2_with_chronic":
        roi_dir = os.path.join(base_path, "Results", "Intermediate_Files", "mz3_files", f"{model}_{type_prediction}")


        
    #Get list of nifti files for a given model and type of prediction
    roi_files = [f for f in os.listdir(roi_dir) if f.endswith(".nii.gz")]

    # Extract MNI coordinates
    results = []

    for roi_file in sorted(roi_files):
        path = os.path.join(roi_dir, roi_file)
        roi_nii = nib.load(path)
        roi_data = roi_nii.get_fdata()

        if np.sum(roi_data) == 0:
            results.append((roi_file.replace("_mask.nii.gz", ""), "Empty"))
            continue

        # Compute center of mass in voxel space
        com_voxel = center_of_mass(roi_data)
        com_mni = coord_transform(*com_voxel, roi_nii.affine)

        roi_name = roi_file.replace("_mask.nii.gz", "")
        results.append((roi_name, np.round(com_mni, 2)))

    # Print results
    print("\n=== ROI Center MNI Coordinates ===\n")

    return results


###############################################SIGNIFICANCE TESTS###############################################################

def permutation_test(null_perf, actual_perf, combined_parcels=False):
    ''' 
    Computes p_values comparing the mean fold-level performance per iteration in the null distribution to
    the actual mean fold-level performance of the iteration for the original data
    
    Params:
    null_perf: dataframe containing the performance (roc score) of the classifier on the empirical null distribution
    actual_perf: dataframe containing the performance (roc score) of the classifier on the real distribution

    Returns:
    actual_perf dataframe containing the n_extreme and p_value. n_extreme is the number of times the null observation
    exceeds the observation, and p_value is the empirical p_value calculated through permutation test.
    '''
    
    # Initialise column of n_extreme (number of times a null distribution exceeds the real observation)
    actual_perf["n_extreme"] = 0

    #Initialise column for stat metric empirical p-value calculated with permutation test
    actual_perf["p_value"] = 0.0

    #Ensure index of the real and null dataframes start from 0 for comparison
    actual_perf = actual_perf.reset_index(drop=True)
    null_perf = null_perf.reset_index(drop=True)

    #Make a list of mean roc auc values per parcel (over folds for each iteration -- if multiple, one assumed) of the real distribution
    actual_perf["mean_over_folds"] = actual_perf["performance_stats"].apply(
        lambda x: [np.mean(fold_scores) for fold_scores in x["individual_scores"]][0]
        )

    #Make a list of mean roc auc values per parcel (over folds for each iteration) of the null distribution
    null_perf["mean_over_folds"] = null_perf["performance_stats"].apply(
    lambda x: [np.mean(fold_scores) for fold_scores in x["individual_scores"]]
    )

    for i, value in enumerate(actual_perf["mean_over_folds"]):
        true_score = actual_perf.at[i, "mean_over_folds"]
        null_scores = null_perf.at[i, "mean_over_folds"] 

        #Calculate number of times that the null observations are more extreme than the real observations
        n_extreme = sum(score >= true_score for score in null_scores)
        n_iterations = len(null_scores)

        #Normalise n_extreme by the number of iterations to get a p-value
        p_val = (n_extreme) / (n_iterations)

        actual_perf.at[i, "n_extreme"] = n_extreme
        actual_perf.at[i, "p_value"] = p_val

    return actual_perf


# def visualize_top_rois(avg_parcel_accuracies, std_parcel_accuracies, combined_labels, combined_img, top_n=5):
#     # Sort the ROIs by their accuracies and select the top n
#     sorted_rois = sorted(avg_parcel_accuracies.items(), key=lambda item: item[1], reverse=True)
#     top_rois = sorted_rois[:top_n]
    
#     # Decode labels if they are byte literals
#     top_labels = [roi[0].decode("utf-8") if isinstance(roi[0], bytes) else roi[0] for roi in top_rois]
#     top_accuracies = [roi[1] for roi in top_rois]

#     # Safely retrieve standard deviations, ignoring labels not found in std_parcel_accuracies
#     top_std = [std_parcel_accuracies.get(label, 0) for label in top_labels]  # Default to 0 if label missing
    
#     # Create a bar graph of the top n ROIs and their accuracies with error bars
#     plt.figure(figsize=(12, 8))
#     plt.barh(top_labels, top_accuracies, xerr=top_std, color='skyblue')
#     plt.xlabel('ROC AUC Score')
#     plt.title(f'Top {top_n} ROIs by Accuracy')
#     plt.gca().invert_yaxis()
#     plt.show()
    
#     # Plot the top 20 ROIs on a glass brain
#     combined_data = combined_img.get_fdata()
#     mask_data = np.zeros_like(combined_data)
    
#     # Map labels to indices
#     label_to_index = {label.decode("utf-8") if isinstance(label, bytes) else label: i + 1 for i, label in enumerate(combined_labels)}
    
#     # Apply mask for top_labels and assign unique values
#     for label in top_labels:
#         if label in label_to_index:
#             idx = label_to_index[label]
#             mask_data[combined_data == idx] = idx
    
#     # Create a colormap with 20 unique colors
#     colors = plt.cm.get_cmap('tab20', top_n)
#     cmap = ListedColormap(colors(range(top_n)))
    
#     masked_img = nib.Nifti1Image(mask_data, combined_img.affine)
    
#     # Plot the combined atlas with only the top 20 ROIs
#     plotting.plot_roi(masked_img, title=f'Top {top_n} ROIs in Combined Atlas', display_mode='ortho', draw_cross=False, cmap=cmap)
#     plotting.show()

#     # Plot the top n ROIs on a glass brain centered at x=0
    
#     plotting.plot_glass_brain(masked_img, title=f'Top {top_n} ROIs on Glass Brain (x=0)', display_mode='lyrz', cut_coords=(0, 0, 0), colorbar=True, cmap=cmap)
    
#     # Create custom legend with top ROIs
#     legend_patches = [Patch(color=colors(i), label=top_labels[i]) for i in range(top_n)]
#     plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)  # Move legend outside plot area
#     plt.show()





# def visualize_top_rois(avg_parcel_accuracies, std_parcel_accuracies, combined_labels, combined_img, top_n=20):
#     '''
#     Function that visualises the top_n ROIS output by a classifier, ordered by accuracy
#     '''
#     # Sort the ROIs by their accuracies and select the top 20
#     sorted_rois = sorted(avg_parcel_accuracies.items(), key=lambda item: item[1], reverse=True)
#     top_rois = sorted_rois[:top_n]
    
#     top_labels = [roi[0] for roi in top_rois]
#     top_accuracies = [roi[1] for roi in top_rois]
#     top_std = [std_parcel_accuracies[label] for label in top_labels]
    
#     # Create a bar graph of the top 20 ROIs and their accuracies with error bars
#     plt.figure(figsize=(12, 8))
#     plt.barh(top_labels, top_accuracies, xerr=top_std, color='skyblue')
#     plt.xlabel('ROC AUC Score')
#     plt.title('Top 20 ROIs by Accuracy')
#     plt.gca().invert_yaxis()
#     plt.show()
    
#     # Plot the top 20 ROIs on a glass brain
#     combined_data = combined_img.get_fdata()
#     mask_data = np.zeros_like(combined_data)
    
#     # Map labels to indices
#     label_to_index = {label: i + 1 for i, label in enumerate(combined_labels)}
    
#     # Apply mask for top_labels and assign unique values
#     for label in top_labels:
#         if label in label_to_index:
#             idx = label_to_index[label]
#             mask_data[combined_data == idx] = idx
    
#     # Create a colormap with 20 unique colors
#     colors = plt.cm.get_cmap('tab20', top_n)
#     cmap = ListedColormap(colors(range(top_n)))
    
#     masked_img = nib.Nifti1Image(mask_data, combined_img.affine)
    
#     # Plot the combined atlas with only the top 20 ROIs
#     plotting.plot_roi(masked_img, title='Top 20 ROIs in Combined Atlas', display_mode='ortho', draw_cross=False, cmap=cmap)
#     plotting.show()

#     # Plot the top 20 ROIs on a glass brain centered at x=0
#     plotting.plot_glass_brain(masked_img, title='Top 20 ROIs on Glass Brain (x=0)', display_mode='lyrz', cut_coords=(0, 0, 0), colorbar=True, cmap=cmap)
#     plotting.show()



# def get_schaefer_parcellation(n_rois=200):
#     '''
#     Function that fetches Schaefer atlas and returns images and labels
#     of parcels
#     '''
#     # Fetch Schaefer atlas
#     schaefer_parc = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2, verbose=False)
#     schaefer_img = nib.load(schaefer_parc['maps'])
#     print("schaefer")
#     return schaefer_img, schaefer_parc['labels'].tolist()



# def get_tian_parcellation(tian_atlas_path, tian_label_path):
#     '''
#     Function that loads the Tian Atlas and returns images and labels
#     of parcels
#     '''
#     # Load Tian atlas
#     tian_img = nib.load(tian_atlas_path)
    
#     # Load Tian labels
#     with open(tian_label_path, 'r') as f:
#         tian_labels = [line.strip() for line in f.readlines()]
    
#     return tian_img, tian_labels



def load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path):
 
    #Open rating files as it contains the mapping to run number per visit 
    run_mapping_df = pd.read_csv(run_mapping_path, delimiter = '\t')

    #Here we pick run_02 in case a visit has both run1 and run2, choose keep = 'first' to pick run_01 instead
    run_mapping_df = run_mapping_df.drop_duplicates(subset = ['subject', 'visit'], keep = 'last')

    #Select the run of choice (run_mapping_df) contains either run_02 or run_01 when available for a visit
    subject_label = f"sub-{subject_id:03d}"
    visits_col = run_mapping_df[run_mapping_df["subject"] == subject_label]["visit"]
    runs_col = run_mapping_df[run_mapping_df["subject"] == subject_label]["run"]
    runs = list(zip(visits_col.tolist(), runs_col.tolist()))

    file_paths = generate_file_paths(subject_id, visit_range, z_map_path, run_map = runs)
    
    visit_data = []
    
    for path in file_paths:
        if os.path.exists(path):
            nifti_img = nib.load(path)
            visit_data.append(nifti_img)
        else:
            return None  # Return None if any visit is missing
    
    # Return the list of NIfTI images for the visits
    if len(visit_data) == 4:
        return visit_data
    else:
        return None
    


#Making function to make a selected_subjects_labels.csv for the case including chronic
def select_with_chronic_labels(second_level_glm):

    second_level_glm = pd.read_csv(second_level_glm)

    selected_subjects_df = pd.DataFrame({
        "subject_id": second_level_glm["Unnamed: 0"] 
    })
    selected_subjects_df["subject_id"] = [int(s.split("-")[1]) for s in selected_subjects_df["subject_id"]]

    conditions = [
        second_level_glm["chronic"] == 1,
        second_level_glm["SBPp"] == 1,
        second_level_glm["SBPr"] == 1
    ]

    choices = [-1, 1, 0]

    selected_subjects_df["label"] = np.select(
        conditions,
        choices,
        default=np.nan
    )

    selected_subjects_df["label"] = [int(x) for x in selected_subjects_df["label"]]

    # critical step
    selected_subjects_df = selected_subjects_df.dropna()

    print(selected_subjects_df["label"].value_counts())

    selected_subjects_df.to_csv("with_chronic_selected_subjects_labels.csv", index=False)

    return selected_subjects_df


def select_sbpp_chronic_from_second_level_glm(second_level_glm_path):
    """Function that takes the second_level_glm csv file path as input
    and returns a csv file of a dataframe excluding SBPr patients
    second_level_glm paths"""

    #Filter subjects that belong to the SBP and chronic groups, excluding SBPr
    second_level_glm_df = pd.read_csv(second_level_glm_path)

    second_level_glm_no_sbpr = second_level_glm_df[
        (second_level_glm_df["chronic"] == 1) | 
        (second_level_glm_df["SBPp"] == 1)
    ]

    #Proving that the filtering works, you should get array([0])
    display(second_level_glm_no_sbpr["SBPr"].unique())

    #Count how many participants are SBPp and SBPr, since there is no SBPr, here SBPp is 1 and chronic is 0
    display(second_level_glm_no_sbpr["SBPp"].value_counts())

    second_level_glm_no_sbpr.to_csv("second_level_glm_no_sbpr_csv.csv")
    


######################                  LATER NOTEBOOK

# def combine_parcellations(schaefer_img, tian_img, schaefer_labels, tian_labels): 
#     ''' 
#     Function that combines the Schaefer and Tian parcellations and returns 
#     combined images and labels
#     '''
#     # Resample Tian atlas to match the Schaefer resolution 
#     tian_resampled = image.resample_to_img(tian_img, schaefer_img, interpolation='nearest')
    
#     # Get data arrays
#     schaefer_data = schaefer_img.get_fdata()
#     tian_data = tian_resampled.get_fdata()
    
#     # Increment Tian atlas labels to avoid overlap
#     tian_data[tian_data > 0] += schaefer_data.max()
    
#     # Combine the maps into a single NIfTI image
#     combined_data = np.copy(schaefer_data)
#     combined_data[tian_data > 0] = tian_data[tian_data > 0]
#     combined_img = nib.Nifti1Image(combined_data, schaefer_img.affine)

#     # Combine the labels
#     combined_labels = schaefer_labels + tian_labels
#     print("combined parcellations")
#     return combined_img, combined_labels

# def combine_parcellations(schaefer_img, tian_img, schaefer_labels, tian_labels):
#     ''' 
#     Function that combines the Schaefer and Tian parcellations and returns 
#     combined images and labels
#     '''
#     # Resample Tian atlas to match the Schaefer resolution 
#     tian_resampled = image.resample_to_img(tian_img, schaefer_img, interpolation='nearest')
    
#     # Get data arrays
#     schaefer_data = schaefer_img.get_fdata()
#     tian_data = tian_resampled.get_fdata()
    
#     # Increment Tian atlas labels to avoid overlap
#     tian_data[tian_data > 0] += schaefer_data.max()
    
#     # Combine the maps into a single NIfTI image
#     combined_data = np.copy(schaefer_data)
#     combined_data[tian_data > 0] = tian_data[tian_data > 0]
#     combined_img = nib.Nifti1Image(combined_data, schaefer_img.affine)

#     # Combine the labels
#     combined_labels = schaefer_labels + tian_labels
    
#     return combined_img, combined_labels


# def get_combined_parcellation_for_X(X, tian_atlas_path, tian_label_path, n_rois=200):
#     schaefer_img, schaefer_labels = get_schaefer_parcellation(n_rois)
#     tian_img, tian_labels = get_tian_parcellation(tian_atlas_path, tian_label_path)
#     combined_img, combined_labels = combine_parcellations(schaefer_img, tian_img, schaefer_labels, tian_labels)

#     # Define the masker and extract time series
#     nlb = NiftiLabelsMasker(combined_img, strategy="mean")

#     R_combined = nlb.fit_transform(X)
#     return R_combined, combined_labels

# def run_combined_parcellation_classification(X, S_all, tian_atlas_path, tian_label_path, iterations = 100,
#                                              freq_printing = 10, nr_PCA_components = 40, n_splits = 5):
#     '''
#     Function that runs the classification on the combined parcels obtained from Tian and 
#     Schefer atlases.  
#     '''
#     accuracies = []
#     for i in range(iterations):
#         if i % freq_printing == 0:
#             print("iter", i, end='; ')
#         # We re-initialize these objects for clarity
#         scaler = RobustScaler()
#         logisticClassifier = LogisticRegression()
#         pca = PCA(n_components=nr_PCA_components)
#         skf = StratifiedKFold(n_splits, shuffle=True)

#         # The make_pipeline function returns a Pipeline object
#         pipe = make_pipeline(scaler, pca, logisticClassifier)
#         # print(pipe)

#         R_combined, _ = get_combined_parcellation_for_X(X, tian_atlas_path, tian_label_path)

#         folds = skf.split(R_combined, S_all)

#         # And now cross-validate *all* the steps
#         acc_av_todo = 0
#         for i, fold in enumerate(folds):
#             # Here, we unpack fold (a tuple) to get the train and test indices
#             train_idx, test_idx = fold
#             test_idx = test_idx.astype(int)
#             pipe.fit(R_combined[train_idx], S_all[train_idx])

#             preds = pipe.predict(R_combined[test_idx])
#             acc_av_todo += roc_auc_score(S_all[test_idx], preds)  # originally accuracy score, but need to account for the class imbalance

#         acc_av_todo /= n_splits
#         accuracies.append(acc_av_todo)

#     return accuracies