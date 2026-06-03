# Moment-to-Moment Pain Variability Predicts Pain Chronification
This study aims to investigate whether short-term variability in reported pain ratings can be leveraged to predict pain chronification. To do this, we take data from Baliki et al. (2012) comprising a set of participants with subacute back pain (SBP) and two control groups, namely a healthy and a chronic group. The participants were followed longitudinally over the course of one year, divided in 4 visits. At each visit, they reported spontaneous pain ratings while in an fMRI scanner, providing continuous pain ratings as well as neural activity. After one year, SBP participants are classified as either recovering or persistent based on whether their average reported pain dropped by 20%. 

# Full paper:
Moment-to-Moment Pain Variability Predicts Pain Chronification
Gaia Pantaleo, Carl Ashworth, Maeghal Jain, Flavia Mancini
bioRxiv 2026.03.02.709010; doi: https://doi.org/10.64898/2026.03.02.709010

## Pre preprocessing step using fmriprep
If this is a new download of the dataset (!IMPORTANT: ONLY RUN IF THIS IS A NEW DOWNLOAD OF THE DATASET), then run `05_fmriprep_corrections.ipynb`, add the `.bidsignore` file to the dataset, and convert all SamplingFrequency in dataset directory from string to float. This will generate the output file needed to run preprocessing using notebooks (If this does not work, see the bottom of this readme `"Checking if the dataset is bids compliant"`).

Example code to run pre-preprocessing using fmriprep:
singularity run --cleanenv fmriprep-22.0.0.simg openpain.org/subacute_longitudinal_study/ output/ participant --participant-label 101 --nthreads 16 --verbose --fs-license-file license.txt

### Order to run analysis:
1. First, run preprocessing files. Details about the files, are shown in (Preprocessing: Notebooks Summary). *Note this step is very computationally intensive and should be run on a cluster.*
2. Run downsampling and hierarchical GLM to obtain z-scores summarising neural correlates associated to pain variability (details are in Downsampling, regression and GLMs: Notebooks Summary).
3. Run classification: predict whether participants will recover or progress to persistent pain(details in Classification: Notebooks Summary). 
*Note: To allow testing the code structure we have included some pre-obtained results in this project, which are needed to run classification. These include the atlases (Tian and Schaefer, which can be found in the directory `melbourne atlas`); the design matrix for the second level GLM `secondlevelGLMdf.csv` which contains the classification labels and sessions attended, and the resampled first level z maps `Carl_first_level_z_maps_resampled_mask_2` for one participant.*
4. Run pain ratings variability analysis (see below, Ratings analysis: Notebooks Summary). 

## Preprocessing: Notebooks Summary
Notebooks starting in with a number perform initial steps of dataset handling. They include preliminary investigations of the dataset followed by preprocessing then cleaning fMRI data. These files can be run in numerical order. 

NOTE: This is a computationally intensive step.

1. `01_basic_analysis` Basic dataset checks, seeing what's present and whether it is BIDs compliant.
2. `02_missing_data` Analysing how many visits, files are present for each participant.
4. `04_func_data_and_resp_length` Fixes issues regarding reponse length of the functional data found in `01_basic_analysis`.
5. `05_fmriprep_corrections.ipynb` Makes changed to the source task-related files (JSON files) to  make the dataset BIDS complaint.
6. `06_analysing_chronic_patients` Narrow down the participants that can be analysed.
7. `07_preproc_check.ipynb` Check to ensure success of the preprocessing of the files.
8. `09_preprocess_and_resample_all_files` preprocesses and resamples all files.

## Downsampling, regression and GLMs: Notebooks and Files Summary
Notebooks starting in `Carl_` detail the response smoothing, downsampling, regression and GLMs. detail the response smoothing, downsampling, regression and GLMs. These notebooks can be run in numerical order.

The "preprocessed_responses" files which get generated when running these notebooks will be in a `Carl_preprocessed_responses` folder, and will contain:

- `Carl_preprocessed_responses/`
    - convolved_lag_1_differences.csv
    - convolved_movement_regressors.csv
    - convolved_responses.csv
    - lag_1_differencces.csv
    - movement_regressors.csv
    - preprocessed_response_details.csv
    - responseArray.csv
    - unsmoothedResponseArray.csv

The functions related to the GLMs and response preprocessing are located in:
- `Carl_Response_Functions` -> This is the central location for the functions used by for hierarchical GLM related tasks.

The output of the GLM will be in:
- `Carl_second_level_z_maps_09_05_2025/`
    - chronic_baseline
    - error
    - movement
    - pain


## Classification: Notebooks Summary
Notebooks starting in `classification` detail the classification of prediction of classes labels in visit 4 based on the z-scores from visit 1. z-scores are output by the first level GLM. These can be run in numerical order.
`classification_01_prediction_and_permutation_test` performs classification using both GLM and SVM algorithms starting from visit 1 to predict visit 4. Input to the algorithms is z-scores output by hierarchical GLMs, therefore preprocessing and GLMs are two steps that have to be performed before running this notebook. Classification with GLM takes about 1 hour on CPU, SVM about 30 mins. Output is a csv file saved to a location which will be specified in the relevant classification block once it is run.
Permutation of the ROIs labels is used to ensure statistical validity. It is also present in this notebook, by default it is 1000 iterations. Then, checks to ensure that permutation was performed correctly, as well as the location of the csv file containing the permutation results are saved to a location output in the relevant block. Runtime for the SVM classifier is about 2 hours, while for GBM about 3.5 hours.

NOTE: In our final analysis, we only consider GBM output due to the non-linear nature of the data, which make it a better predictor compared to linear models as SVM. We only keep SVM prediction here as extra analysis confirming that non-linear models are more appropriate for our problem.

NOTE: It is important to run the display_top_ROIs_from_classifier() and visualize_top_rois_from_performance_df() as these are part of the analysis of the dataframe, not just visualisation functions. They are constrained to always output a plot to ensure double-checking of the output.

`classification_02_visit2_predicts_visit4` detail the classification of prediction of classes labels in visit 4 based on the z-scores from visit 2. Runtimes are comparable to classification_01.

`classification_03_FDR_correction` performs FDR correction both using the Benjamini Hochberg and Storey's Q value methods (both for SVM and GBM output, but our analysis only focuses on GBM; SVM is performed for completeness). The file plots the resulting performance of the algorithm after false discovery rate correction and saves the results to a CSV file, which location will be output by the block.

`classification_04_figures` produces figure 2 and 3 on the paper, and supplementary figures. It uses the CSV output of the FDR correction, so classification and FDR needs to be performed at least once, but not necessarily more than once, to produce the figures.

`classification_05_predict_SBP_chronic` performs supplementary analysis in which the algorithm needs to predict the risk of chronification between persistent pain participants and chronic pain participants. It performs the equivalent of classification_01 and classification_02 on this set of participants. Here, we only use GBM. Runtimes are comparable to classification_01.

## Ratings analysis: Notebooks Summary
Notebooks startig in `ratings_` detail analysis, preprocessing, missing data exclusion and statistical analysis of the pain ratings. These can be run in numerical order.

`ratings_01_variability analysis` cleans missing data in case participants have less than 30% of reported ratings (ratings=0 consecutively for over 30% of the session). Ratings are also demeaned here and then analysis on variability of the ratings is performed and plotted.

`ratings_02_statistical_analysis` performs statistical analysis on ratings for validation of results.


## ORIGINAL DATASET
### Paper and Dataset

1. (paper)
"Corticostriatal functional connectivity predicts transition to chronic back pain", Baliki et al. (2012), Nature Neuroscience, https://www.nature.com/articles/nn.3153

2. OpenPain where dataset associated to paper is publicly available:
https://www.openpain.org/


### Dataset Summary
Summary:
5475 Files, 109.26GB
122 - Subjects
5 - Sessions

Available Tasks:
Matched visual stimulus
Rating spontaneous pain 1
Resting state
Standard visual stimulus

Available Modalities:
fMRI

### Checking if the dataset is bids compliant
1. These steps were performed to preprocess dataset. This step does not need to be performed more than once.
    + Run the file: `01_basic_analysis.ipynb`
    + If you've downloaded the dataset for the first time then the `participants.json` file in the dataset isn't validated. You will have to add an extra bracket at line 126. Use https://jsonlint.com to validate your json files. 
    + If you've downloaded the dataset for the first time then the `sessions.json` file in the  dataset isn't validated. You will have to add an extra bracket at line 276. Use https://jsonlint.com to validate your json files. 
    + Section 1.1 of the file: `01_basic_analysis.ipynb` runs basic checks, to ensure that the setup of the user is correct. It should pass if jupyter is correctly installed and the right environment (please see below) is correctly set.
4. Missing data handling:
    + If fMRI data was insufficient in a session, that session was deleted. If there was an alternative session (run 2) that session was preprocessed and kept if valid. If the pain ratingsin a specific session were equal to 0 for 30% of the session, the session was deleted.

5. FMRIPrep Installation
    + Followed this guide: https://www.nipreps.org/apps/singularity/ and checked the docker version using: https://hub.docker.com/r/nipreps/fmriprep/
    + If this is a new download of the dataset (!IMPORTANT: ONLY RUN IF THIS IS A NEW DOWNLOAD OF THE DATASET), then run `05_fmriprep_corrections.ipynb`, add the `.bidsignore` file to the dataset, and convert all SamplingFrequency in dataset directory from string to float.
    + Then create an fmriprep singularity container and then shifting it to HPC for running. 
    WARNING: It is possible that it will not run completely because it will consume excessive CPU time on the login nodes.

### Requirements for running code 
1. Setup the  environment to run the notebooks: details about the environment requirements are in environment.yml

2. Please note that on each of the files in the first block you can find a "user" variable. Please change that to the name of the user based on the machine it is run on (e.g. "user x")
WARNING: If user is not set, the paths to existing files, and paths to generate files will be incorrect. 

