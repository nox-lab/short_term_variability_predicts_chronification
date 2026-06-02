import numpy as np
from numpy.fft import rfftfreq, rfft
import scipy
from scipy.signal import welch, sosfiltfilt, butter, savgol_filter
from scipy.stats import norm as normal_dist
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from Response_Functions.load_responses import *
from Response_Functions.plotting import generate_hrf
import os
import pandas as pd
from typing import Callable

def downsample_response(response, get_freqs = False):
  """
  If `get_freqs = False` returns a downsampled array of length 244.
  If `get_freqs = True` returns the presumed sampling rate of the response given as input. 

  It does this by simplying sampling, if the response length is a multiple of 8784.
  However if it is not, then it has different rules.
  """
  length = len(response)
  if length == 244:
    downsample_response = response
    fs = float(1.0/2.5)
  elif length % 8784 == 0:
    downsample_response = response[::(length//8784)*36]
    fs = (length//8784) * 14.4
  elif length == 2196:
    downsample_response = response[::9]
    fs = 7.2
  elif length == 16711:
    print(f"I can't deal with this response length: {length}")
    raise ValueError(f"Bad response length {length}")
  elif 8784-length > 36:
    response_full = np.zeros(8784)
    response_full[:length] = response
    response_full[length:] = response[-1]
    downsample_response = response_full[::36]

    fs = 14.4
  else:
    downsample_response = response[:8784][::36]
    fs = 14.4

  if get_freqs:
    return fs
  else:
    return downsample_response

def low_pass_filter_response(subject,visit, run,response, fs=14.4, plot=True, plot_extra=False):
  """
  Low Pass filters a response.

  Takes `response` - the response to low pass filter.

  Returns the low-pass filtered response.
  """

  n = len(response)
  freq = rfftfreq(n, d=1./fs)
  freqs, psd = welch(response,fs=fs)
  psd = np.abs(psd)
  power_percentages = np.cumsum(psd)/np.sum(psd)
  second = savgol_filter(power_percentages,11,4,2)
  cutoff_freq = freqs[np.where(second > 0-0.02)[0][0]]
  if cutoff_freq == 0:
    print("defaulting to 0.4 Hz cutoff")
    cutoff_freq = 0.4

  if plot_extra:
    plt.plot(freqs,power_percentages)
    plt.axvline(cutoff_freq, linestyle='dashed')
    plt.show()

    plt.plot(freqs, second, label='second derivative of power percentages')
    plt.axvline(cutoff_freq, linestyle='dashed')

    plt.show()

  sos = butter(10, cutoff_freq, 'lp', fs=fs, output='sos')
  filtered = sosfiltfilt(sos, response)
  
  if plot_extra:
    times = np.arange(n)*2.5/36
    FFT = rfft(response)

    fig, ax = plt.subplots(2,figsize=(10,6),dpi=500)
    ax[0].plot(times[4*36:],response[4*36:],label = 'Unsmoothed')
    # ax[0].set_xticks(times[4*36::36*8])
    # ax[0].set_xticklabels(np.arange(len(times[4*36::36]*36)))
    ax[0].set_xlabel("Seconds",fontsize=15)
    ax[0].set_ylabel("Pain Rating",fontsize=15)
    ax[1].plot(freqs,power_percentages,label='Cumulative PSD')
    ax[1].axvline(cutoff_freq, 0,1,linestyle='dashed',label='cutoff freq',color='red')
    ax[1].set_xlabel("Frequency (Hz)",fontsize=15)
    ax[1].set_ylabel("Signal Power %",fontsize=15)
    ax[1].legend()
    ax[0].plot(times[4*36:],filtered[4*36:],label='Smoothed')
    ax[0].legend()
    plt.suptitle(f"The Smoothed and Unsmoothed Response of subject {subject[-3:]}, visit {visit[-1]}",fontsize=15)
    plt.tight_layout()
    plt.show()
  elif plot:
    times = np.arange(n)*2.5/36
    FFT = rfft(response)

    fig, ax = plt.subplots(1,figsize=(20,2))
    ax.plot(times,response)
    ax.plot(times,filtered)
    plt.suptitle(f"The smoothed and un-smoothed response of subject {subject[-3]}, {run}, visit {visit[-1]}")
    plt.tight_layout()
    plt.xlabel("seconds")
    plt.ylabel("pain rating")
    plt.show()

  return filtered

def compute_lag_1_differences(array):
  """ Computes the differences between successive values in an array row, prepending 0."""
  differences = -np.diff(array,1, prepend=array[:,0][np.newaxis].T)
  return differences

def calc_motion_regressor(responses,plot_subject: list, responses_df, threshold,plot=False):
  """
  Returns a movement regressor of the same length as responses, by setting the value to 1 
  whenever the absolute change in response is above `threshold`.
  """
  #compute the movement regressors
  subject = responses_df.iloc[plot_subject]['subject'].values[0]
  mov_regressor = np.zeros_like(responses)
  print(mov_regressor.shape)
  mov_regressor[:,1:] = np.diff(responses,1)

  mov_regressor = np.where(np.abs(mov_regressor) > threshold, 1, 0)
  if plot:
    fig, ax = plt.subplots(2,1,figsize=(20,4),sharex=True)
    ax[1].plot(mov_regressor[plot_subject,:].T, label='movement regressor')
    ax[0].plot(norm(responses[plot_subject,:]).T, label='normalised response')
    ax[1].legend(fontsize=15)
    ax[0].legend(fontsize=15)
    plt.xlabel("seconds", fontsize=15)
    ax[0].set_ylabel("pain rating", fontsize=15)
    plt.suptitle(f"Movement Regressors for {subject}, change threshold= {threshold}",fontsize=20)
    plt.show()

  return mov_regressor

def conv(array_row, hrf, doNorm=True, returnLength = 8784):
  """ Convolve a response with a hrf. Returns a vector of the same length as the input array."""
  if doNorm:
    convolution = np.convolve(norm(array_row),hrf)
  else:
    convolution = np.convolve(array_row,hrf)
  return convolution[:returnLength]


def calculate_residual_variances(path, sub=6):
  """
  Plots residual variances and deviations for a response array compared to a smoothed response array, where each row is a response.
  """

  smoothedResponseArray = readCsv("responseArray.csv",path=path)
  unsmoothedResponseArray = readCsv("unsmoothedResponseArray.csv",path=path)
  residuals = unsmoothedResponseArray-smoothedResponseArray
  print(residuals.shape)
  stds = np.std(smoothedResponseArray,axis=1)
  residual_deviations = np.std(residuals,axis=1)

  resids = residuals[:,:].flatten()
  print("Kurtosis", kurtosis(resids))
  print(len(resids))
  space = np.linspace(-20,20,2000)
  plt.hist(resids,bins=1000,log=False,density=True,label='Histogram of Residuals')
  plt.plot(space,normal_dist.pdf(space,0,np.std(resids)),label='Best Fit Gaussian')
  print(kurtosis(np.random.normal(0, np.std(resids), size=10_000)))
  #plt.hist(np.random.normal(0, np.std(resids), size=10_000),density=True,bins=500, label='equivalent normal')
  plt.title("The response residuals ")
  plt.xlabel("Residual Value")
  #plt.xlim(np.min(resids),np.max(resids))
  plt.xlim(-5,5)
  plt.legend()
  plt.show()

  plt.title("The distribution of the \nresidual standard deviations/smoothed standard deviations")
  plt.xlabel("Residual Standard Deviation")
  plt.ylabel("Number of Subjects")
    # Assuming residual_deviations and stds are your data arrays
  filtered_values = residual_deviations[stds != 0] / stds[stds != 0]

  plt.hist(filtered_values, bins=80)
  plt.show()



def low_pass_filter_and_downsample_response(response):
  """
  Performs low pass filtering and downsampling on a single response.

  Returns the downsampled response as an array

  """
  fs = downsample_response(response, get_freqs=True)
  if len(response) == 244:
    downsampled_response = response
  else:
    smoothed_response = low_pass_filter_response(None,None,None,response, fs, plot=False, plot_extra=False)
    downsampled_response = downsample_response(smoothed_response,get_freqs=False)

  return np.array(downsampled_response)

def save_preprocessed_responses(preprocessedResponseArray, unmodifiedResponseArray, preprocessedResponseDetails_df, saveDirectory):
  """
  Takes the outputs of the response preprocessing, and a save directory, and saves the outputs to that directory.
  returns nothing.

  This function will stop and return if the directory already exists, thus preventing any overwriting.
  """
  if os.path.exists(saveDirectory):
    #import shutil
    #shutil.rmtree(saveDirectory)
    print(f"Old directory exists {saveDirectory}")
    print("Please use a new folder or delete the old one.")
    return
  else:
    os.makedirs(saveDirectory)
    print(f"Created directory {saveDirectory}")

  responses_path = os.path.join(saveDirectory, 'responseArray.csv')
  DF = pd.DataFrame(preprocessedResponseArray)
  DF.to_csv(responses_path,sep='\t',index=False, header=False)

  responses_path_2 = os.path.join(saveDirectory, 'unsmoothedResponseArray.csv')
  DF2 = pd.DataFrame(unmodifiedResponseArray)
  DF2.to_csv(responses_path_2,sep='\t',index=False, header=False)

  responses_details_path = os.path.join(saveDirectory, 'preprocessed_response_details.csv')
  preprocessedResponseDetails_df.to_csv(responses_details_path, sep='\t', index=False,header=True)


def preprocess_all_responses(preprocessing_function: Callable[[np.ndarray],np.ndarray],cut=4) -> 'tuple[np.ndarray,np.ndarray, pd.DataFrame]':
  """
  Loads and iterates over all of the chronic and subacute responses, and calls the preprocessing function on each response array, before returning the 
  downsampled original responses, new preprocessed response array, and a pandas df containing the details of the generated arrays.

  The function `preprocessing_function` should taking a single np.ndarray response and return the preprocessed response.
  """

  # First get the chronic and subacute dataframes
  participants_df = pd.read_csv('openpain.org/subacute_longitudinal_study/participants.tsv', sep='\t')
  chronic_and_subacute_df = participants_df.loc[(participants_df["group"] == "chronic") | (participants_df["group"] == "subacute")]
  # Create an array to save the responses to, and a pandas df to store the visit, run, subject, and chronic/subacute
  responseArray = []
  unsmoothedResponseArray = []
  responseDetails = []
  
  # Now go through each subject and smooth response for run 1 and each visit using a smoothing method
  response_lengths = []
  for index, row in chronic_and_subacute_df.iterrows():
    subject = row['participant_id']
    group = row['group']
    responses_one_participant, key = get_all_responses_for_one_subject(subject,'sp',downsampled = False, verbose = False)
    for i, key in enumerate(key):
      visit, run = key.split(', ')
      #!Removed if statement excepting run 2

      response = responses_one_participant[i]
      
      # Attempt to preprocess the response
      try:
        preprocessed_response = preprocessing_function(response)

        responseArray.append(preprocessed_response)
        responseDetails.append([subject, visit, run, group])
        unsmoothedResponseArray.append(downsample_response(response))
      except Exception as err:
        plt.plot(response)
        plt.show()
        print(err, subject, visit, run)

  #now convert the response lists to arrays.
  responseArray = np.array(responseArray)
  responseArray[responseArray < 0] = 0

  unsmoothedResponseArray = np.array(unsmoothedResponseArray)
  
  fig = plt.figure(figsize=(20,10))
  plt.plot(responseArray.T[:,::16])
  plt.plot(unsmoothedResponseArray.T[:,::16],linestyle='dotted')
  plt.show()

  responseDetails = pd.DataFrame(responseDetails, columns=['subject','visit','run','group'])

  responseArrayCut = responseArray[:,cut:]
  unsmoothedResponseArrayCut = unsmoothedResponseArray[:,cut:]

  return responseArrayCut,unsmoothedResponseArrayCut, responseDetails


def generate_and_save_convolved_regressors(path = "Carl_preprocessed_responses/"): 
  """Loads the regressors from `path`/, convolves them, and then saves them back to `path/` as `convolved_{regressor}`"""
  
  if os.path.isfile(os.path.join(path, 'convolved_movement_regressors.csv')) or os.path.isfile(os.path.join(path, 'convolved_lag_1_differences.csv')) or os.path.isfile(os.path.join(path, 'convolved_responses.csv')):
    print(f"Old regressor files exist here")
    print("Please ensure these files do not already exist before running this code.")
    return
 
  regressors = {
    'responses':readCsv('responseArray.csv', path = path),
  'lag_1_differences':readCsv('lag_1_differences.csv', path = path),
  'movement_regressors':readCsv('movement_regressors.csv', path = path),
  }

  for key, item in regressors.items():
    print(f'shape of {key}: {item.shape}')


  hrf = generate_hrf(2.5,1,plot=False)

  convolved_regressors = {
    'responses': np.zeros_like(regressors['responses']),
    'lag_1_differences': np.zeros_like(regressors['responses']),
    'movement_regressors': np.zeros_like(regressors['responses'])
    }
  
  for i in range(regressors['responses'].shape[0]):
    convolved_regressors['responses'][i,:] = conv(regressors['responses'][i,], hrf, True, regressors['responses'].shape[1])
    convolved_regressors['lag_1_differences'][i,:] = conv(regressors['lag_1_differences'][i,:], hrf, True, regressors['responses'].shape[1])
    convolved_regressors['movement_regressors'][i,:] = conv(regressors['movement_regressors'][i,:], hrf, False, regressors['responses'].shape[1])

  for key, array in convolved_regressors.items():
    DF = pd.DataFrame(array)
    save_path = os.path.join(path, f'convolved_{key}.csv')
    DF.to_csv(save_path,sep='\t',index=False, header=False)
   

def generate_regressors(path):
  """ Generates and saves the unconvolved regressors by using the response arrays at `path/`, without convolving them. Saves unconvolved regressors to `path/`"""

  responseArray = readCsv("responseArray.csv",path=path)
  responses_df = readCsv("preprocessed_response_details.csv", path=path, header_and_df=True)
  if os.path.isfile(os.path.join(path, 'movement_regressors.csv')) or os.path.isfile(os.path.join(path, 'lag_1_differences.csv')):
    print(f"Old regressor files exist here")
    print("Please use a new folder or delete the old one.")
    return

  regressors = {
    'movement_regressors': calc_motion_regressor(responseArray[:,:],[0], responses_df, threshold= 1),
    'lag_1_differences': compute_lag_1_differences(responseArray),
  }
  print(responseArray.shape)

  for key, regressor in regressors.items():
    save_path = os.path.join(path, f'{key}.csv')
    DF = pd.DataFrame(regressor)
    DF.to_csv(save_path,sep='\t',index=False, header=False)

