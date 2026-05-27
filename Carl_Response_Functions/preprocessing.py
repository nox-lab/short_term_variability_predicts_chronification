import os
import shutil
import nibabel as nb
import warnings

import seaborn as sns

import imageio
from nilearn import plotting, image
from IPython.display import SVG

sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
warnings.filterwarnings('ignore')

def load_chronic_patients_list():
    """
    Load the list of chronic patients from the text file chronic_patients.txt
    This list is collated in the file: 06_analysing_chronic_patients.ipynb
    
    Parameters
    ----------
    None
    Returns
    -------
    chronic_patients : list[str]
    """
    # f = open("chronic_patients.txt")
    chronic_patients = "097,098,099,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122".split(",")
    return chronic_patients

def get_anat_file_paths(subject_id, visit_id):
    """
    Get the anat file paths (both not preprocessed and preprocessed) of the patient data
    
    Parameters
    ----------
    subject_id : int, str
        Should be an id of a chronic patient only, as they are only preprocessed now.
        These ids can be obtained from load_chronic_patients_list()
    visit_id : int, str
        visit id, should be valid for the patient taken
    Returns
    -------
    anat_file_not_processed : str
    anat_file_processed : str
    """
    # 9/10/22 changes: processed doesn't need to be compared to the unprocessed file
    # instead the template file and the preprocessed file need to be compared.
    # anat_file_not_processed = "openpain.org/subacute_longitudinal_study/sub-" + str(subject_id) + "/ses-visit" + str(visit_id) + "/anat/sub-" + str(subject_id) + "_ses-visit" + str(visit_id) + "_T1w.nii.gz"
    anat_file_processed = "output/sub-" + str(subject_id) + "/anat/sub-" + str(subject_id) + "_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    
    # anat_file_template = "/home/mj606/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
    # anat file template changed to: "/home/mj606/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"
    anat_file_template = "/home/mj606/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"
    return anat_file_processed, anat_file_template

def compare_two_files(file_1, file_2, file_1_description = "preprocessed", file_2_description = "template"):
    """
    Create images and a gif from those images to compare the results of preprocessing.
    Output is stored in "preproc_check_output/".
    
    Parameters
    ----------
    file_1 : str
    file_2 : str
    Returns
    -------
    gif_path : str
        path where the gif is stored
    """
    anat_loaded_file_1 = nb.load(file_1)
    anat_loaded_file_2 = nb.load(file_2)
    plot_file_1 = plotting.plot_epi(anat_loaded_file_1, display_mode="mosaic", title=file_1_description, cmap="gray")
    plot_file_2 = plotting.plot_epi(anat_loaded_file_2, display_mode="mosaic", title=file_2_description, cmap="gray")
    
    preproc_output_dir = "preproc_check_output/"
    images_sub_dir = "images/"
    gifs_sub_dir = "gifs/"
    
    image_file_1 = preproc_output_dir + images_sub_dir + file_1.split("/")[1] + "_" + file_1_description + ".png"
    image_file_2 = preproc_output_dir + images_sub_dir + file_2_description + ".png"
    gif_path =  preproc_output_dir + gifs_sub_dir + file_1.split("/")[1] + "_comparison.gif"
    
    print(image_file_1, image_file_2, gif_path)
    try:
        os.makedirs(preproc_output_dir + images_sub_dir)
    except:
        pass
    
    try:
        os.makedirs(preproc_output_dir + gifs_sub_dir)
    except:
        pass
    
    plot_file_1.savefig(image_file_1)
    plot_file_2.savefig(image_file_2)
    images = [imageio.imread(image_file_1), imageio.imread(image_file_2)]
    imageio.mimsave(gif_path, images, duration=2)
    return gif_path

def make_comparison_gifs_for_all_chronic_patients():
    """
    Loop through all chronic patients and create the comparison gifs.
    Output is stored in "preproc_check_output/".
    
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    chronic_patients = load_chronic_patients_list()
    all_gif_paths = []
    not_done_for = []
    for chronic_patient in chronic_patients:
        print("loading patient " + chronic_patient + "...", sep="")
        anat_file_processed, anat_file_template = get_anat_file_paths(chronic_patient, 1)
        try:
            gif_path = compare_two_files(anat_file_processed, anat_file_template)
            all_gif_paths.append(gif_path)
        except:
            not_done_for.append(chronic_patient)
            print("not done for ", chronic_patient)
    return all_gif_paths, not_done_for


def get_bbregister_file_path(subject_id, visit_id):
    """
    Get the bbregister_file_path of patient
    
    Parameters
    ----------
    subject_id : int, str
        Should be an id of a chronic patient only, as they are only preprocessed now.
        These ids can be obtained from load_chronic_patients_list()
    visit_id : int, str
        visit id, should be valid for the patient taken
    Returns
    -------
    bbregister_file_path : str
    """
    bbregister_file_path = "output/sub-" + str(subject_id) + "/figures/sub-" + str(subject_id) + "_ses-visit" + str(visit_id) + "_task-mv_desc-bbregister_bold.svg"
    return bbregister_file_path

def get_all_bbregister(parent_dir = "output"):
    """
    get registeration files in one folder
    """
    chronic_patients = load_chronic_patients_list()
    try:
        os.mkdir("bbregister")
    except:
        pass
    for chronic_patient in chronic_patients:
        print("loading patient " + chronic_patient + "...", sep="")
        for visit_id in range(1, 5):
            print("loading visit " + str(visit_id) + "...", sep="")
            bbregister_file_path = get_bbregister_file_path(chronic_patient, visit_id)
            if os.path.exists(bbregister_file_path):
                output_path = "bbregister/" + "/".join(bbregister_file_path.split("/")[3:])
                shutil.copyfile(bbregister_file_path, output_path)

