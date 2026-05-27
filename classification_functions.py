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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import matplotlib.image as mpimg
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import ast
import datetime
import json
from Hendrix_utils import *

# --- Prediction per ROI --- 

def save_classification_results_to_json(subjects_data, S_all, classifier, classifier_name, tian_atlas_path, tian_label_path,
                                         early_prediction, visit_2, visit_3, with_chronics = False,
                                         nr_iterations=1, n_rois=200, top_n=5, setup_style='default', 
                                         base_path ="base path", distribution="Real", visit_range=range(1, 5)):
    """
    Runs the classification and saves the results to a JSON file.
    """

    accuracies = []
    scaler = RobustScaler()

    combined_data = []
    combined_labels = []
    
    for subject_data in subjects_data:
       
        R_combined, labels = get_combined_parcellation_for_X(subject_data, tian_atlas_path, tian_label_path, n_rois,
                                                              early_prediction = early_prediction, visit_2 = visit_2,
                                                              visit_3 = visit_3)
        combined_data.append(R_combined)
        if not combined_labels:
            combined_labels = labels
    
    combined_data = np.vstack(combined_data)
    
    # Initialize dictionaries with string keys
    parcel_accuracies = {label.decode() if isinstance(label, bytes) else label: [] for label in combined_labels}
    confusion_matrices = {label.decode() if isinstance(label, bytes) else label: [] for label in combined_labels}
    
    #Initialise dictionary where the performance statistics will be stored
    performance_stats = {}

    for iteration in range(nr_iterations):  
        if iteration % 10 == 0:
            print(f"Iteration {iteration}")
        
        for parcel_idx, label in enumerate(combined_labels):
            if isinstance(label, bytes):
                label = label.decode()  # Convert label from bytes to string

            parcel_data = combined_data[:, parcel_idx].reshape(-1, 1)

            # Create pipeline with the specified classifier
            pipe = make_pipeline(scaler, classifier)

            if label not in parcel_accuracies:
                print(f"Warning: Label {label} was not found in parcel_accuracies. Skipping.")
                continue

            #Cross validation
            scores = []
            all_true = []
            all_pred = []
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = iteration)
            for train_idx, test_idx in skf.split(parcel_data, S_all):
                pipe.fit(parcel_data[train_idx], S_all[train_idx])

                #Calculate class predictions
                preds = pipe.predict(parcel_data[test_idx])

                #Calculate continuous confidence score for model used in ROC AUC computation
                if hasattr(pipe, "predict_proba"):
                    probs = pipe.predict_proba(parcel_data[test_idx])[:, 1]
                else:
                    probs = pipe.decision_function(parcel_data[test_idx])

                if len(np.unique(S_all[test_idx])) < 2:
                    print(f"Warning: Only one class present in y_true for fold. Skipping this fold.")
                    continue

                # Collect scores
                scores.append(roc_auc_score(S_all[test_idx], probs))

                # Compute optimal threshold using Youden's J statistic
                fpr, tpr, thresholds = roc_curve(S_all[test_idx], probs)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx]

                # Apply optimal threshold to get binary predictions
                preds = (probs >= optimal_threshold).astype(int)

                # Collect predictions for confusion matrix
                all_true.extend(S_all[test_idx])
                all_pred.extend(preds)

                # Compute confusion matrix per fold
                cm = confusion_matrix(S_all[test_idx], preds)
                confusion_matrices[label].append(cm.tolist())
        
            # Store the ROC AUC scores for each iteration
            parcel_accuracies[label].append(scores)

    for label in parcel_accuracies:
        scores_array = np.array(parcel_accuracies[label])
        performance_stats[label] = {
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'max': np.max(scores_array),
            'min': np.min(scores_array),
            'individual_scores': scores_array.tolist()
            }

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   
    results = {
        'setup_info': {
            'setup_style': setup_style,
            'date': date,
            'classifier': classifier_name,
            'visit_range': list(visit_range),
        },
        'performance_stats': ensure_string_keys(performance_stats),
        'confusion_matrices': ensure_string_keys(confusion_matrices)
    }

    # Create a filename based on the setup style, date, classifier, and visit range
    if early_prediction and not with_chronics:
        filename = f"{classifier_name}_reproduced_early_prediction.json"
    
    if with_chronics and early_prediction:
        filename = f"{classifier_name}_with_chronics_early_prediction.json"
    
    elif visit_2 and not with_chronics:
        filename = f"{classifier_name}_reproduced_visit_2.json"

    elif visit_2 and with_chronics:
        filename = f"{classifier_name}_with_chronics_visit_2.json"

    elif visit_3:
        filename = f"{classifier_name}_reproduced_visit_3.json"

    else:
        filename = f"{classifier_name}_reproduced_all_visits.json"

    filepath = os.path.join(generate_output_classifier_dir(base_path, classifier_name, f"{distribution}", "fMRI"), filename)

    # Save the results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {filepath}")

    return filepath


# Main function to run the classifier and save the results
def main(classifier, visit_range, classifier_name, z_map_path, run_mapping, S_all, base_path,
          tian_atlas_path, tian_label_path, visit_selection = "all", early_prediction = False,
          visit_2 = False, visit_3 = False,
          distribution = "Real",
          selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"):


    # Ensure there are at least 25 subjects in each class
    selected_subjects = []
    selected_labels = []

    if visit_selection == "all":
        # Identify subjects classified as 0 and 1
        subjects_class_0 = []
        subjects_class_1 = []
        for subject_id in range(1, len(S_all)):  # Loop through all subjects
            if S_all[subject_id - 1] == 0:
                subjects_class_0.append(subject_id)
            elif S_all[subject_id - 1] == 1:
                subjects_class_1.append(subject_id)

        for subject_id in subjects_class_0:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(0)

            
        for subject_id in subjects_class_1:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(1)

    
        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "1_and_4":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)
        

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit1', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()
       

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_2":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        # selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit2', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    elif visit_selection == "visit_3":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        # selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit3', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    # Convert combined data to a consistent 2D array
    selected_labels = np.array(selected_labels[:len(valid_subjects)])  # Adjust labels to match the valid subjects

    if visit_selection == "1_and_4":
        early_prediction = True

    if visit_selection == "visit_2":
        visit_2 = True

    if visit_selection == "visit_3":
        visit_3 = True
    
    # Run the classification and save the results
    results_file_path = save_classification_results_to_json(
        subjects_data, selected_labels, classifier, classifier_name, tian_atlas_path, tian_label_path, early_prediction, visit_2, visit_3,
        base_path = base_path, distribution = distribution)




# --- Combined predictive power of all ROIs ----

def combined_parcels_classification(subjects_data, labels, classifier, base_path, tian_atlas_path, tian_label_path,
                                    classifier_name = "GBM", early_prediction = "all", n_rois = 200, n_iterations=1,
                                    distribution = "Real"):
    """ 
    Function that runs a classification model on all combined parcels. It outputs the filepath where the performance
    results are stored in the form of a dictionary. By default it runs classification on the real data. If param 
    distribution is set to "Null", it runs classification on an empirical null distribution. 
    """

    scaler = RobustScaler()

    scores = []
    combined_data = []
    combined_labels = []
    
    for subject_data in subjects_data:
       
        R_combined, parcel_labels = get_combined_parcellation_for_X(subject_data, tian_atlas_path, tian_label_path, n_rois, early_prediction = early_prediction)
        combined_data.append(R_combined)

    combined_data = np.vstack(combined_data)

    for iteration in range(n_iterations):

        if distribution == "Null":
            #Shuffle the labels, seed in each iteration is fixed for reproducibility 
            labels = np.random.RandomState(iteration).permutation(labels)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=iteration)

        for train_idx, test_idx in skf.split(combined_data, labels):
            if classifier_name == "GBM":
                pipe = make_pipeline(RobustScaler(), SelectKBest(f_classif, k=50),
                                  PCA(n_components=50), classifier)
            else:
                pipe = make_pipeline(RobustScaler(), SelectKBest(f_classif, k=50),
                                  PCA(n_components=5), classifier)
                
            pipe.fit(combined_data[train_idx], labels[train_idx])

            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(combined_data[test_idx])[:, 1]
            else:
                probs = pipe.decision_function(combined_data[test_idx])

            scores.append(roc_auc_score(labels[test_idx], probs))

    result = ensure_string_keys({
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores),
                'individual_scores': scores
                })
    
  
    if early_prediction:
        filename = f"{classifier_name}_combined_parcels_early_prediction.json"
    else:
        filename = f"{classifier_name}_combined_parcels_all_visits.json"

    filepath = os.path.join(generate_output_classifier_dir(base_path, classifier_name, data_type="fMRI",
                                                            distribution= distribution),
                                                            filename)
  

    # Save the results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4)

    return filepath


def run_combined_parcels_classification(classifier, classifier_name, z_map_path, run_mapping,
                                        S_all, base_path, tian_atlas_path, tian_label_path,
                                        visit_selection ="all", visit_range=range(1,5),
                                        nr_iterations = 1, early_prediction = False,
                                        distribution = "Real"):
    
    # Ensure there are at least 25 subjects in each class
    selected_subjects = []
    selected_labels = []

    if visit_selection == "all":
        # Identify subjects classified as 0 and 1
        subjects_class_0 = []
        subjects_class_1 = []
        for subject_id in range(1, 55):  # Loop through all subjects
            if S_all[subject_id - 1] == 0:
                subjects_class_0.append(subject_id)
            elif S_all[subject_id - 1] == 1:
                subjects_class_1.append(subject_id)

    
        for subject_id in subjects_class_0:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)

            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(0)
            
        for subject_id in subjects_class_1:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(1)


        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "1_and_4":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit1', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    # Convert combined data to a consistent 2D array
    selected_labels = np.array(selected_labels[:len(valid_subjects)])  # Adjust labels to match the valid subjects

    if visit_selection == "1_and_4":
        early_prediction = True
        
    # Run the classification and save the results
    results_file_path = combined_parcels_classification(subjects_data, selected_labels, classifier, base_path,
                                                        tian_atlas_path, tian_label_path, 
                                                        classifier_name = classifier_name,
                                                        early_prediction = early_prediction,
                                                        n_rois = 200, n_iterations=nr_iterations,
                                                        distribution = distribution)
        


# --- Classification per ROI on empirical null distribution --- 

def save_classification_permutation_results_to_json(subjects_data, S_all, classifier, classifier_name, tian_atlas_path, tian_label_path,
                                                    early_prediction, visit_2, visit_3,
                                                    iterations, with_chronics = False, base_path = "base path",
                                                    n_rois=200, top_n=5, setup_style='default', visit_range=range(1, 5)):
    """
    Runs the classification and saves the results to a JSON file.
    """

    accuracies = []
    scaler = RobustScaler()

    combined_data = []
    combined_labels = []
    for subject_data in subjects_data:
        R_combined, labels = get_combined_parcellation_for_X(subject_data, tian_atlas_path, tian_label_path, n_rois)
        combined_data.append(R_combined)
        if not combined_labels:
            combined_labels = labels
    
    combined_data = np.vstack(combined_data)
    
    # Initialize dictionaries with string keys
    parcel_accuracies = {label.decode() if isinstance(label, bytes) else label: [] for label in combined_labels}
    confusion_matrices = {label.decode() if isinstance(label, bytes) else label: [] for label in combined_labels}

    for iteration in range(iterations):  
        if iteration % 10 == 0:
            print(f"Iteration {iteration}")
            
        #Shuffle the labels, seed in each iteration is fixed for reproducibility 
        shuffled_S_all = np.random.RandomState(iteration).permutation(S_all)
        
        for parcel_idx, label in enumerate(combined_labels):
            if isinstance(label, bytes):
                label = label.decode()  # Convert label from bytes to string

            parcel_data = combined_data[:, parcel_idx].reshape(-1, 1)

            # Create pipeline with the specified classifier
            pipe = make_pipeline(scaler, classifier)

            # Use cross_val_score for parallel cross-validation
            #scores = cross_val_score(pipe, parcel_data, shuffled_S_all, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc', n_jobs=16)

            # Ensure we are always using string labels
            if label not in parcel_accuracies:
                print(f"Warning: Label {label} was not found in parcel_accuracies. Skipping.")
                continue

            # Store the ROC AUC scores for each iteration
            #parcel_accuracies[label].append(scores)
            
            #COMMENT: Again here we were running k-fold cross validation twice, fixed it
            all_true = []
            all_pred = []
            scores = []

            # Compute confusion matrix for each fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = iteration)
            for train_idx, test_idx in skf.split(parcel_data, shuffled_S_all):
                pipe.fit(parcel_data[train_idx], shuffled_S_all[train_idx])
                preds = pipe.predict(parcel_data[test_idx])

                if hasattr(pipe, "predict_proba"):
                    probs = pipe.predict_proba(parcel_data[test_idx])[:, 1]
                else:
                    probs = pipe.decision_function(parcel_data[test_idx])

                if len(np.unique(shuffled_S_all[test_idx])) < 2:
                    print(f"Warning: Only one class present in y_true for fold. Skipping this fold.")
                    continue

                scores.append(roc_auc_score(shuffled_S_all[test_idx], probs))

                all_true.extend(shuffled_S_all[test_idx])
                all_pred.extend(preds)

                cm = confusion_matrix(shuffled_S_all[test_idx], preds)
                confusion_matrices[label].append(cm.tolist())  # Convert to list for JSON compatibility

            parcel_accuracies[label].append(scores)
                
    # Compute mean, std, max, min values for each parcel
    performance_stats = {}
    for label in parcel_accuracies:
        scores = np.array(parcel_accuracies[label])
        performance_stats[label] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'max': np.max(scores),
            'min': np.min(scores),
            'individual_scores': scores.tolist()
        }

    # Store setup information (e.g., classifier, visit range, date)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    classifier_name = type(classifier).__name__

    results = {
        'setup_info': {
            'setup_style': setup_style,
            'date': date,
            'classifier': classifier_name,
            'visit_range': list(visit_range),
        },
        'performance_stats': ensure_string_keys(performance_stats),
        'confusion_matrices': ensure_string_keys(confusion_matrices)
    }

    # Create a filename based on the setup style, date, classifier, and visit range
    if early_prediction and not with_chronics:
        filename = f"{classifier_name}_reproduced_early_prediction_null.json"

    elif with_chronics and early_prediction:
        filename = f"{classifier_name}_with_chronics_early_prediction_null.json"

    elif visit_2 and not with_chronics:
        filename = f"{classifier_name}_reproduced_visit_2_null.json"
    
    elif with_chronics and visit_2:
        filename = f"{classifier_name}_with_chronics_visit_2_null.json"

    elif visit_3:
        filename = f"{classifier_name}_reproduced_visit_3_null.json"

    else:
        filename = f"{classifier_name}_reproduced_all_visits_null.json"


    filepath = os.path.join(generate_output_classifier_dir(base_path, classifier_name, "Null", "fMRI"), filename)

    # Save the results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {filepath}")

    return filepath


def null_main(classifier, visit_range, classifier_name, z_map_path, run_mapping, S_all, base_path, 
              tian_atlas_path, tian_label_path, visit_selection = "all", early_prediction = False,
               visit_2 = False, visit_3 = False, iters = 1000):

    # Identify subjects classified as 0 and 1
    subjects_class_0 = []
    subjects_class_1 = []

    for subject_id in range(1, 55):  # Loop through all subjects
        if S_all[subject_id - 1] == 0:
            subjects_class_0.append(subject_id)
        elif S_all[subject_id - 1] == 1:
            subjects_class_1.append(subject_id)

    # Ensure there are at least 25 subjects in each class
    selected_subjects = []
    selected_labels = []

    if visit_selection == "all":
    
        for subject_id in subjects_class_0:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(0)
        for subject_id in subjects_class_1:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(1)


        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "1_and_4":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit1', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_2":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit2', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_3":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit3', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([0, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    # Convert combined data to a consistent 2D array
    selected_labels = np.array(selected_labels[:len(valid_subjects)])  # Adjust labels to match the valid subjects

    if visit_selection == "1_and_4":
        early_prediction = True

    if visit_selection == "visit_2":
        visit_2 = True

    if visit_selection == "visit_3":
        visit_3 = True
  
    # Run the classification and save the results
    results_file_path = save_classification_permutation_results_to_json(
        subjects_data, selected_labels, classifier, classifier_name, tian_atlas_path, tian_label_path, 
        early_prediction=early_prediction, base_path = base_path, visit_2 = visit_2, visit_3 = visit_3,
        iterations = iters)


def with_chronic_null_main(classifier, visit_range, classifier_name, z_map_path, run_mapping, S_all, base_path, 
              tian_atlas_path, tian_label_path, visit_selection = "all", early_prediction = False,
               visit_2 = False, visit_3 = False, selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv",
               iters = 1000):

    # Identify subjects classified as 0 and 1
    subjects_class_0 = []
    subjects_class_1 = []

    for subject_id in range(1, 55):  # Loop through all subjects
        if S_all[subject_id - 1] == 1:
            subjects_class_0.append(subject_id)
        elif S_all[subject_id - 1] == -1:
            subjects_class_1.append(subject_id)

    # Ensure there are at least 25 subjects in each class
    selected_subjects = []
    selected_labels = []

    if visit_selection == "all":
    
        for subject_id in subjects_class_0:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(0)
        for subject_id in subjects_class_1:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(1)


        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "1_and_4":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit1', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([1, -1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_2":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit2', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([1, -1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_3":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit3', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([-1, 1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    # Convert combined data to a consistent 2D array
    selected_labels = np.array(selected_labels[:len(valid_subjects)])  # Adjust labels to match the valid subjects

    if visit_selection == "1_and_4":
        early_prediction = True

    if visit_selection == "visit_2":
        visit_2 = True

    if visit_selection == "visit_3":
        visit_3 = True
  
    # Run the classification and save the results
    results_file_path = save_classification_permutation_results_to_json(
        subjects_data, selected_labels, classifier, classifier_name, tian_atlas_path, tian_label_path, 
        early_prediction=early_prediction, with_chronics = True, base_path = base_path, visit_2 = visit_2, visit_3 = visit_3,
        iterations = iters)
    
    
def no_recovery_main(classifier, visit_range, classifier_name, z_map_path, run_mapping, S_all, base_path,
          tian_atlas_path, tian_label_path, visit_selection = "all", early_prediction = False,
          visit_2 = False, visit_3 = False,
          distribution = "Real",
          selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"):


    # Ensure there are at least 25 subjects in each class
    selected_subjects = []
    selected_labels = []

    if visit_selection == "all":
        # Identify subjects classified as 0 and 1
        subjects_class_0 = []
        subjects_class_1 = []
        for subject_id in range(1, len(S_all)):  # Loop through all subjects
            if S_all[subject_id - 1] == 1:
                subjects_class_0.append(subject_id)
            elif S_all[subject_id - 1] == -1:
                subjects_class_1.append(subject_id)

        for subject_id in subjects_class_0:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(0)

            
        for subject_id in subjects_class_1:
            data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
            if data is not None:
                selected_subjects.append(subject_id)
                selected_labels.append(1)

    
        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "1_and_4":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)
        

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit1', 'visit3'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([1, -1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()
       
       
        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)


    elif visit_selection == "visit_2":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        # selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit2', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([1, -1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    elif visit_selection == "visit_3":

        #Match the rating df to see select ids and labels by visit (only 1 and 4)
        rate_of_change_csv_path = "Results/Intermediate_Files/painrating_rate_of_change.csv"
        # selected_labels_csv_path = "Results/Intermediate_Files/selected_subjects_labels.csv"

        # Load data
        rate_of_change_data = pd.read_csv(rate_of_change_csv_path)
        #Keeping only run_02 in case there is both a run_01 and a run_02 for the same subject and visit
        rate_of_change_data = rate_of_change_data.drop_duplicates(subset=['subject', 'visit'], keep ='last')
        labels_df = pd.read_csv(selected_labels_csv_path)

        # Adjust 'subject' to match 'subject_id'
        rate_of_change_data['subject'] = rate_of_change_data['subject'].str.replace('sub-', '').astype(int)

        chosen_subjects = (
            rate_of_change_data[rate_of_change_data['visit'].isin(['visit3', 'visit4'])]
            .groupby('subject')['visit']
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )

        labels_df = labels_df[labels_df['label'].isin([1, -1])]
        labels_df['subject_id'] = labels_df['subject_id'].astype(int)

        # Filter labels only for subjects that have both visits
        labels_df = labels_df[labels_df['subject_id'].isin(chosen_subjects)]

        # Now extract lists
        selected_subjects = labels_df['subject_id'].tolist()
        selected_labels = labels_df['label'].tolist()

        # Ensure both classes are present
        if len(np.unique(selected_labels)) < 2:
            print("Error: Need at least two classes for classification.")
        else:
            # Prepare data for the selected subjects
            subjects_data = []
            valid_subjects = []

            for subject_id in selected_subjects:
                data = load_and_aggregate_subject_data(subject_id, visit_range, z_map_path, run_mapping_path = run_mapping)
                if data is not None:
                    subjects_data.append(data)
                    valid_subjects.append(subject_id)

    # Convert combined data to a consistent 2D array
    selected_labels = np.array(selected_labels[:len(valid_subjects)])  # Adjust labels to match the valid subjects

    if visit_selection == "1_and_4":
        early_prediction = True

    if visit_selection == "visit_2":
        visit_2 = True

    if visit_selection == "visit_3":
        visit_3 = True
    
    # Run the classification and save the results
    results_file_path = save_classification_results_to_json(
        subjects_data, selected_labels, classifier, classifier_name, tian_atlas_path, tian_label_path, early_prediction, visit_2, visit_3,
        base_path = base_path, with_chronics = True, distribution = distribution)
