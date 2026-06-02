from bids import BIDSLayout
import json
import pandas as pd
import os

import gzip
import matplotlib.pyplot as plt
import numpy as np

from nilearn import image as nimg
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
sns.set_theme(style='whitegrid', palette='bright', color_codes=True)

def norm(x):
  return (x - np.mean(x))

def readCsv(file_path):
  path = "Carl_preprocessed_responses/"
  return np.array(pd.read_csv(os.path.join(path,file_path), delimiter='\t', header=None))

def get_all_visits_for_one_subject(participant):
    """
    returns a list of ints representing the visits that the participant was at, using the sessions.tsv file
    """
    session_file = f"openpain.org/subacute_longitudinal_study/{participant}/{participant}_sessions.tsv"
    f = pd.read_csv(session_file, sep='\t')
    return [i[-1] for i in list(f['session_id'].loc[f['session_id'].str.match('visit*')])]

def get_all_responses_for_one_subject(subject, task_type,downsampled=True, verbose = True):
    """Gets all the pain responses across sessions and runs for one subject."""
    responses = []
    key = []
    
    visits = get_all_visits_for_one_subject(subject)
    if verbose:
        print(subject, visits)
    
    for visit in visits:
        for run in [1,2]:
            try:
                response = load_single_subject_response(subject,visit,run,plot=0,task_type=task_type, display = False,downsample=downsampled)
                responses.append(response)
                key.append(f'visit{visit}, run_0{run}')
            except Exception as err:
                if verbose:
                    print(err)
                continue
            
            
    return np.array(responses, dtype='object'),key 

def plot_all_responses_for_one_subject(subject, task_type,downsampled = False, fs = 14.40):

    fig, ax = plt.subplots(figsize=(15,6))

    responses, key = get_all_responses_for_one_subject(subject, task_type, downsampled = downsampled)

    for response in responses:
        ax.plot([i/fs for i in range(len(response))], response) # downsizing responses
    plt.ylim(0, 110)
    plt.xlabel('seconds')
    plt.legend(key)
    plt.title(f"Pain ratings across runs for subject {subject} on the {task_type} task, downsampled = {downsampled}")
    return responses,key
      
def load_single_subject_response(subject: str, visit: int, run: int, plot: int = 0,task_type = 'sp', display = True,downsample=True):
    """
    Loads a single subject response from a task within a run and visit, and optionally plots it
    
    Parameters
    ----------
    participant_id: str
        Examples: "sub-001", "sub-002", ... "sub-122".
        Range: ["001", "122"]
    
    visit: int
        Examples: 1, 2, .. 5
        Range: [1, 5] but for some participants it's [1, 4]
    
    run: int
        Examples: 1, 2
        Range: [1, 2]
    
    plot: Bool
        Plots data if True, else does not
       
    task_type: str
        Examples: "sp","sv","mv"
        defaults to "sp"
    
    Returns
    -------
    downsampled response : numpy.ndarray
    
    else throws an error stating the reponse that could not be found
    """
    participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')
    
    if display:
        print("loading the data of: ")
        print(participants_df.loc[participants_df['participant_id'] == subject, ["group", "race", "gender", "age", "origin"]])
    session = "visit" + str(visit)

    if task_type == 'sp':
        resp_file = f"openpain.org/subacute_longitudinal_study/{subject}/ses-visit{visit}/func/{subject}_ses-visit{visit}_task-sp_run-0{run}_resp.tsv.gz"
    else:
        resp_file = f"openpain.org/subacute_longitudinal_study/{subject}/ses-visit{visit}/func/{subject}_ses-visit{visit}_task-{task_type}_resp.tsv.gz"
        
    try:
        f = gzip.open(resp_file, 'rb')
    except:
        err = f"resp not available, subject {subject}, visit {visit}, run {run}"
        raise ValueError(err)

    response = f.read().decode("utf-8").split("\n")
    to_delete = []
    for i in range(0, len(response)):
        try:
            response[i] = float(response[i])
        except:
            to_delete.append(i)

    for i in to_delete:
        del response[i]

    if (len(response) != 244) and downsample:
        response = response[:8784:36] 
    if display:
        print("len of responses: ", len(response))
    if plot:
        fig, ax = plt.subplots(figsize=(16,6))
        plt.title(f"{'Downsampled' if downsample else ''} Responses")
        ax.plot(response)
    return np.array(response)


def read_json(fname):
    """ Takes the JSON file containing the descriptions of the tasks, and prints them."""
    f = open (fname, "r")
    json_file = json.loads(f.read())
    print(json.dumps(json_file, indent = 1))


def norm(x):
  return (x - np.mean(x))

def readCsv(file_path,path = "Carl_preprocessed_responses/",header_and_df: bool = False) :
  """ 
  Loads either a numpy array without header or a pandas df with header, depending on `header_and_df`.
  
  The file is loaded from: `{path}/{filepath}`
  """

  if header_and_df== True:
    return pd.read_csv(os.path.join(path,file_path), delimiter='\t')
  return np.array(pd.read_csv(os.path.join(path,file_path), delimiter='\t', header=None))

def get_number_of_sessions_for_each_subject():
    """
    Returns a dict that contains the number of sessions a participant was present for.
    Here the key = participant_id, and the value = count os sessions.
    """
    participant_and_sessions = {}
    parent_dir = "openpain.org/subacute_longitudinal_study/"
    for folder in os.listdir(parent_dir):
        if folder.startswith("sub"):
            subject_dir = os.path.join(parent_dir, folder)
            count = 0
            for subject_folder in os.listdir(subject_dir):
                if subject_folder.startswith("ses"):
                    count += 1
            participant_and_sessions[subject_dir[-3:]] = count

    participant_and_sessions = dict(sorted(participant_and_sessions.items(), key=lambda kv: kv[0]))
    return participant_and_sessions

def load_sp_func(participant_id: str, visit: int, run: int, participants_df: pd.DataFrame = None,layout: any = None):
    """
    Loads data from sp task
    
    Parameters
    ----------
    participant_id: str
        Examples: "001", "002", ... "122".
        Range: ["001", "122"]
    
    visit: int
        Examples: 1, 2, .. 5
        Range: [1, 5] but for some participants it's [1, 4]
    
    run: int
        Examples: 1, 2
        Range: [1, 2]

    participants_df: DataFrame
        The pandas dataframe containing the participant details(id,race,gender etc)

    layout: any
        The BIDSLayout of the data path (see code below)
    
    Returns
    -------
    
    """
    if not isinstance(participants_df, pd.DataFrame):
        participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')

    if layout is None:
        data_path = "openpain.org/subacute_longitudinal_study/"
        layout = BIDSLayout(data_path)

    print("loading the data of: ")
    print(participants_df.loc[participants_df['participant_id'] == "sub-" + participant_id, ["group", "race", "gender", "age", "origin"]])
    session = "visit" + str(visit)
    func_files = layout.get(subject=participant_id,
                        datatype="func",
                        extension="nii.gz",
                        task="sp",
                        run=run,
                        return_type="file",
                        session=session
                      )
    flag_1 = 0
    try:
        func_file = func_files[0]
        resp_file = os.path.splitext(os.path.splitext(func_file)[0])[0][:-4] + "resp.tsv.gz"
        print("Paths found for func_file and resp_file: ", func_file, resp_file)
    except:
        print("Functional files don't exist")
        flag_1 = 1
    
    if flag_1 == 0:
        func_img = nimg.load_img(func_file)
        print("func_file img shape", func_img.shape)

        flag_2 = 0
        try:
            f = gzip.open(resp_file, 'rb')
        except:
            flag_2 = 1
            print("resp not available")

        if flag_2 == 0:
            response = f.read().decode("utf-8").split("\n")
            to_delete = []
            for i in range(0, len(response)):
                try:
                    response[i] = float(response[i])
                except:
                    to_delete.append(i)

            for i in to_delete:
                del response[i]
            print("len of responses: ", len(response))
            print("range of responses: ", min(response), max(response))
            if len(response) == 244:
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)
            else:
                print("downsampled len of responses:", len(response[::36]))
                plt.title("Downsampled Responses")
                plt.plot(response[::36])
                plt.ylim(0, 100)
                plt.figure()
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)

def load_mv_func(participant_id: str, visit: int, participants_df: pd.DataFrame = None,layout: any = None):
    """
    Loads data from sp task
    
    Parameters
    ----------
    participant_id: str
        Examples: "001", "002", ... "122".
        Range: ["001", "122"]
    
    visit: int
        Examples: 1, 2, .. 5
        Range: [1, 5] but for some participants it's [1, 4]

    participants_df: DataFrame
        The pandas dataframe containing the participant details(id,race,gender etc)
    
    Returns
    -------
    
    """
    if not isinstance(participants_df, pd.DataFrame):
        participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')

    if layout is None:
        data_path = "openpain.org/subacute_longitudinal_study/"
        layout = BIDSLayout(data_path)

    print("loading the data of: ")
    print(participants_df.loc[participants_df['participant_id'] == "sub-" + participant_id, ["group", "race", "gender", "age", "origin"]])
    session = "visit" + str(visit)
    func_files = layout.get(subject=participant_id,
                        datatype="func",
                        extension="nii.gz",
                        task="mv",
                        return_type="file",
                        session=session
                      )
    flag_1 = 0
    try:
        func_file = func_files[0]
        resp_file = os.path.splitext(os.path.splitext(func_file)[0])[0][:-4] + "resp.tsv.gz"
        stim_file = os.path.splitext(os.path.splitext(func_file)[0])[0][:-4] + "stim.tsv.gz"
        print("Paths found for func_file, stim_file and resp_file: ", func_file, stim_file, resp_file)
    except:
        print("Functional files don't exist")
        flag_1 = 1
    
    if flag_1 == 0:
        func_img = nimg.load_img(func_file)
        print("func_file img shape", func_img.shape)

        flag_2 = 0
        try:
            f = gzip.open(resp_file, 'rb')
            f2 = gzip.open(stim_file, 'rb')
        except:
            flag_2 = 1
            print("resp and stim not available")

        if flag_2 == 0:
            response = f.read().decode("utf-8").split("\n")
            stim = f2.read().decode("utf-8").split("\n")
            
            to_delete = []
            to_delete2 = []
            
            for i in range(0, len(response)):
                try:
                    response[i] = float(response[i])
                except:
                    to_delete.append(i)
            for i in to_delete:
                del response[i]
                
            for i in range(0, len(stim)):
                try:
                    stim[i] = float(stim[i])
                except:
                    to_delete2.append(i)
                    
            for i in to_delete2:
                del stim[i]
                
            print("len of responses: ", len(response))
            print("range of responses: ", min(response), max(response))
            print("len of stimuli: ", len(stim))
            print("range of stimuli: ", min(stim), max(stim))
            
            if len(response) == 244:
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)
                plt.figure()
                plt.title("Stimulus")
                plt.plot(stim)
            else:
                print("downsampled len of responses:", len(response[::36]))
                plt.title("Downsampled Responses")
                plt.plot(response[::36])
                plt.ylim(0, 100)
                plt.figure()
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)
                plt.figure()
                print("downsampled len of stimuli:", len(stim[::36]))
                plt.title("Downsampled Stimuli")
                plt.plot(stim[::36])
                plt.figure()
                plt.title("Stimuli")
                plt.plot(stim)

def load_sv_func(participant_id: str, visit: int,participants_df: pd.DataFrame = None,layout: any = None):
    """
    Loads data from sp task
    
    Parameters
    ----------
    participant_id: str
        Examples: "001", "002", ... "122".
        Range: ["001", "122"]
    
    visit: int
        Examples: 1, 2, .. 5
        Range: [1, 5] but for some participants it's [1, 4]
    
    run: int
        Examples: 1, 2
        Range: [1, 2]

    participants_df: DataFrame
        The pandas dataframe containing the participant details(id,race,gender etc)
    
    Returns
    -------
    
    """
    if not isinstance(participants_df, pd.DataFrame):
        participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')

    if layout is None:
        data_path = "openpain.org/subacute_longitudinal_study/"
        layout = BIDSLayout(data_path)

    print("loading the data of: ")
    print(participants_df.loc[participants_df['participant_id'] == "sub-" + participant_id, ["group", "race", "gender", "age", "origin"]])
    session = "visit" + str(visit)
    func_files = layout.get(subject=participant_id,
                        datatype="func",
                        extension="nii.gz",
                        task="sv",
                        return_type="file",
                        session=session
                      )
    flag_1 = 0
    try:
        func_file = func_files[0]
        resp_file = os.path.splitext(os.path.splitext(func_file)[0])[0][:-4] + "resp.tsv.gz"
        print("Paths found for func_file and resp_file: ", func_file, resp_file)
    except:
        print("Functional files don't exist")
        flag_1 = 1
    
    if flag_1 == 0:
        func_img = nimg.load_img(func_file)
        print("func_file img shape", func_img.shape)

        flag_2 = 0
        try:
            f = gzip.open(resp_file, 'rb')
        except:
            flag_2 = 1
            print("resp not available")

        if flag_2 == 0:
            response = f.read().decode("utf-8").split("\n")
            to_delete = []
            for i in range(0, len(response)):
                try:
                    response[i] = float(response[i])
                except:
                    to_delete.append(i)

            for i in to_delete:
                del response[i]
            print("len of responses: ", len(response))
            print("range of responses: ", min(response), max(response))
            if len(response) <= 244:
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)
            else:
                print("downsampled len of responses:", len(response[::36]))
                plt.title("Downsampled Responses")
                plt.plot(response[::36])
                plt.ylim(0, 100)
                plt.figure()
                plt.title("Responses")
                plt.plot(response)
                plt.ylim(0, 100)