import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Response_Functions.load_responses import readCsv, norm
from nilearn.glm.first_level import glover_hrf
from scipy.signal import welch
from nilearn import datasets, image, plotting
from nilearn.image import coord_transform
import nibabel as nib 
from numpy.fft import rfftfreq, rfft

def display_responses(path):
  """Displays a selection of responses from the unsmoothed and smoothed response arrays provided."""
  responseArray = readCsv("responseArray.csv",path=path)
  unsmoothedResponseArray = readCsv("unsmoothedResponseArray.csv",path=path)

  fig = plt.figure(figsize=(20,10))

  times = np.arange(4,responseArray.shape[1]+4)

  colors = ['red', 'blue', 'green', 'orange','brown','purple','black','grey','pink','blueviolet','darkslategrey',
  'darkturquoise',
  'darkviolet',
  'deeppink',
  'deepskyblue',
  'dimgray',
  'dimgrey',
  ]

  for index in np.arange(0,100,8):
    color= colors[index//8]
    _ = plt.plot(times,unsmoothedResponseArray.T[:,index],linestyle='dotted', color=color,linewidth=2)
    _ = plt.plot(times,responseArray.T[:,index], color=color, linewidth=1)

  plt.title("A selection of the pain ratings across chronic and subacute patients.",fontsize=20)

  unsmoothed = mlines.Line2D([], [], color='grey', linestyle='dotted',label='unsmoothed' )
  smoothed = mlines.Line2D([], [], color='grey', linestyle='solid',label='smoothed')
  plt.legend(handles=[unsmoothed, smoothed],fontsize=15)
  plt.xlabel("seconds",fontsize=20)
  plt.ylabel("Pain Rating",fontsize=20)
  plt.show()


def display_response_grid(path):
  """ Display a selection of responses for the appendix of the report, from patients that visited all four sessions."""
  responseArray = readCsv("responseArray.csv",path=path)
  responses_df = readCsv("preprocessed_response_details.csv", path=path, header_and_df=True)

  colours = ['#1f77b4', '#ff7f0e', '#5ca02c', '#1a2728', '#9267bd', '#8c564b', '#FF00FF', '#7f7f7f', '#bcbd22', '#17becf']
  #colours = ['#FFC0CB', '#00FFFF', '#FFD700', '#8A2BE2', '#FFA500', '#008000', '#FF00FF', '#FF4500', '#FF1493', '#1E90FF']

  N = 10
  #subjects = ['sub-006','sub-035','sub-046','sub-073','sub-092','sub-101','sub-120']
  subjects = []
  for sub_num in range(N):
    indexes_and_visits = []
    while(len(indexes_and_visits) != 4):

      number = np.random.randint(0,121,dtype=int)
      subject = f'sub-{number:03}'
      while subject in subjects:
        number = np.random.randint(0,121,dtype=int)
        subject = f'sub-{number:03}'
      indexes_and_visits = responses_df.loc[responses_df['subject'] == subject]['visit']
      if 'visit5' in indexes_and_visits.to_numpy():
        indexes_and_visits = indexes_and_visits[~indexes_and_visits.isin(['visit5'])]
      if len(indexes_and_visits) == 4:
        subjects.append(subject)


  fig, axes = plt.subplots(10, 4, sharex='col', sharey='none',figsize=(10,10))
  ax = axes.flatten()
  index=0
  subjects.sort()
  print(subjects)
  for subject in subjects:
    indexes_and_visits = responses_df.loc[responses_df['subject'] == subject]['visit']
    indexes_and_visits = indexes_and_visits[~indexes_and_visits.isin(['visit5'])]

    for j in range(4):
          ax[index].plot(responseArray[indexes_and_visits.index[j]],color=colours[index //4],linewidth=0.9)
          if index % 4 == 0:
            ax[index].set_ylabel(f'{subject[-3:]}', rotation=90, ha='right', va='center',labelpad=6.0)

          if index < 4:
            ax[index].set_title(f'Visit {j+1}')
          index +=1
  plt.suptitle("A plot showing a random selection of unsmoothed subject ratings, across all four visits",fontsize=15)
  fig.supylabel("Subject Number")
  plt.tight_layout()
  plt.show()

def psd(response,fs=14.4):
  """ Returns the frequencies and psd values of a response: `freq, psd`"""  
  n = len(response)
  freq = rfftfreq(n, d=1./fs)
  freqs, psd = welch(response,fs=fs)
  psd = np.abs(psd)
  return freq, psd


def plot_response_stats(path):
    """Plots histograms comparing the variance of the subacute and chronic subject, and the means."""

    responseArray = readCsv("responseArray.csv",path=path)
    responses_df = readCsv("preprocessed_response_details.csv", path=path, header_and_df=True)
    
    subacute = responses_df[responses_df['group']=='subacute'].index
    chronic = responses_df[responses_df['group']=='chronic'].index
    subacute_deviations = np.std(responseArray[subacute],axis=1)
    subacute_means =np.mean(responseArray[subacute],axis=1)
    chronic_deviations = np.std(responseArray[chronic],axis=1)
    chronic_means = np.mean(responseArray[chronic],axis=1)
    plt.hist(subacute_deviations,bins=40,label='Subacute',color=[[0.3,0.7,0.2]])
    plt.hist(chronic_deviations,bins=40,label='Chronic',color=[[0.5,0.2,0.2]])
    plt.title("A Histogram of the Pain Rating Deviations for \n Subacute and Chronic Patients")
    plt.ylabel("Number of subjects")
    plt.xlabel("Pain Rating Deviation")
    plt.legend()
    plt.show()
    plt.hist(subacute_means,bins=20,label="Subacute")
    plt.hist(chronic_means,bins=20,label="Chronic")
    plt.title("A Histogram of the Pain Rating Means for \n Subacute and Chronic Patients")
    plt.ylabel("Number of subjects")
    plt.xlabel("Pain Rating Mean")
    plt.legend()
    print(np.mean(subacute_deviations))
    print(np.mean(chronic_deviations))

    from scipy.stats import ttest_ind
    t_stat,p_val = ttest_ind(subacute_deviations,chronic_deviations)
    print('t_stat: ', t_stat, 'p_val', p_val)
    t_stat,p_val = ttest_ind(subacute_means,chronic_means)
    print('t_stat: ', t_stat, 'p_val', p_val)

def plot_convolved_regressors(path, subject_num= 12):
  """Plots the convolved and unconvolved regressors, from `path/` """

  unconvolved_regressors = [
  readCsv("responseArray.csv",path=path),
  readCsv("lag_1_differences.csv",path=path),
  readCsv("movement_regressors.csv",path=path),
  ]
  convolved_regressors = {
    'Response':readCsv('convolved_responses.csv',path=path),
  'Difference':readCsv('convolved_lag_1_differences.csv',path=path),
  'Movement':readCsv('convolved_movement_regressors.csv',path=path),
  }
  responses_df = readCsv("preprocessed_response_details.csv", path=path, header_and_df=True)


  color='red'
  fig, ax = plt.subplots(3,1, figsize = (18,8),sharex=True,dpi=500)
  ax = ax.flatten()
  for i, item in enumerate(convolved_regressors.items()):
    key, regressor = item
    if key == 'Movement':
      ax[i].plot(unconvolved_regressors[i][subject_num,:],linewidth=2, linestyle='dotted', label='unconvolved')
    else:
      ax[i].plot(norm(unconvolved_regressors[i][subject_num,:]),linewidth=2, linestyle='dotted',label='unconvolved')
    ax[i].plot(regressor[subject_num,:],label = 'convolved',linewidth=2,color=color)
    ax[i].tick_params(labelsize=15)
    ax[i].set_ylabel(f"\n{key}", fontsize=22)
  plt.xlabel("Volume", fontsize=22)

  unsmoothed = mlines.Line2D([], [], color='blue', linestyle='dotted',label='unconvolved' )
  smoothed = mlines.Line2D([], [], color=color, linestyle='solid',label='convolved')
  fig.legend(handles=[unsmoothed, smoothed],fontsize=20,loc='upper right')
    #ax[i].set_title(key, fontsize=20)

  subject = responses_df.iloc[subject_num][0]
  visit = responses_df.iloc[subject_num][1]
  plt.suptitle(f"Convolved Regressors for the GLM, subject {subject[-3:]}, visit {visit[-1]}", fontsize=25)
  plt.tight_layout()
  plt.show()

  fig = plt.figure(figsize=(20,5))
  plt.plot(unconvolved_regressors[0][subject_num,:])
  plt.plot(convolved_regressors['Response'][subject_num,:])
  plt.xlabel("Volume", fontsize=15)
  plt.ylabel("Pain rating", fontsize=15)
  plt.title("The unconvolved regressor", fontsize=20)
  plt.show()


def generate_hrf(tr, oversampling, plot=False):
    """ Generates and optionally plots a standard HRF. """
    hrf = glover_hrf(2.5,1)
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].plot(hrf,'x')
        ax[0].plot(hrf)
        ax[0].set_xlabel("Seconds")
        ax[0].set_ylabel("HRF value")
        ax[0].set_title("HRF")

        freqs, psdVals = welch(hrf)
        ax[1].set_title("The Spectrum of the HRF")
        ax[1].plot(freqs, psdVals)
        plt.show()
    return hrf


def plot_convolved_and_unconvolved_spectra(path,row=18):
    unconvolved_regressors = [
    readCsv("responseArray.csv",path=path),
    readCsv("lag_1_differences.csv",path=path),
    readCsv("movement_regressors.csv",path=path),
    ]
    convolved_regressors = {
      'Response':readCsv('convolved_responses.csv',path=path),
    'Difference':readCsv('convolved_lag_1_differences.csv',path=path),
    'Movement':readCsv('convolved_movement_regressors.csv',path=path),
    }

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(unconvolved_regressors[0][row])
    ax[0].plot(convolved_regressors['Response'][row])

    freq, PSD_Unconvolved = psd(unconvolved_regressors[0][row])
    freq_2, PSD_convolved = psd(convolved_regressors['Response'][row])

    ax[1].plot(freq,PSD_Unconvolved,label='unconvolved')
    ax[1].plot(freq_2,PSD_convolved,label='convolved')
    ax[1].legend()
    plt.show() 


def labelGraph(rating, averagePain, subject, visit, figsize=(5,2)):
  """ Plots a graph labelled with the average pain, visit, and subject, of a pain rating."""
  plt.figure(figsize=figsize)
  plt.plot(rating)#,figsize=figsize)
  plt.title(f"Average Pain: {np.round(averagePain,2)}, {subject},{visit}")
  plt.xlabel("Time (s)")
  plt.ylabel("Instantaneous Pain")
  plt.show()

def get_plot_low_pain_subjects(responses_df,responses,subjects,plot=True, threshold=5):
  """ Gets and plots the subjects with average pain in visit one below a threshold `threshold`"""
  lowPainSubjects = []
  for subject in subjects:
    visits_df = responses_df.loc[responses_df['subject'] == subject]
    visits_df = visits_df[visits_df['visit'] != 'visit5']

    #now get the best linear fit to that 
    y = visits_df['average_pain']

    try:
      for visit in ['visit1']:
        if(np.array(y)[int(visit[-1])-1] <threshold):
          index = responses_df.loc[(responses_df['subject'] == subject) & (responses_df['visit'] == visit)].index
          if plot:
            labelGraph(responses[index].T, np.array(y)[int(visit[-1])-1],subject, visit)
          lowPainSubjects.append(subject)
    except:
      pass
  return lowPainSubjects


def plot_atlas(atlas_name = 'cort-maxprob-thr25-2mm'):
  atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
  mni_template = datasets.load_mni152_template(1)
  #template_file = mni_template['t1']

  PAG_region = "AAN_PAG_MNI152_1mm_v1p0_20150630.nii"
  PAG = nib.load(PAG_region)

  ho_maxprob_atlas_img = image.load_img(atlas['maps'])
  #print(atlas['maps'])
  #print(ho_maxprob_atlas_img.header)

  #print(np.unique(ho_maxprob_atlas_img.get_fdata()))
  # figure = plt.figure()
  # plotting.plot_roi(ho_maxprob_atlas_img,figure=figure)
  # plt.show()
  # figure=plt.figure()
  # plotting.plot_roi(PAG,figure=figure)
  # plt.show()

  figure = plt.figure()
  PAG_resamp = image.resample_img(PAG,ho_maxprob_atlas_img.affine,interpolation='nearest',target_shape=ho_maxprob_atlas_img.shape)
  #PAG_image = nib.Nifti1Image(PAG_resamp.get_fdata(),affine=PAG_resamp.affine)

  PAG_ROI = image.math_img("49 * img", img=PAG_resamp)

  combined_atlas = image.math_img("img1 + img2", img1=ho_maxprob_atlas_img, img2=PAG_ROI)
  print(np.unique(combined_atlas.get_fdata()))
  atlas['maps'] = combined_atlas
  atlas['labels'] = atlas['labels'] + ['Periaqueductal Gray']
  coord = np.array(coord_transform(0,-32,-10,np.linalg.inv(ho_maxprob_atlas_img.affine))).astype(int)
  print(coord)
  # coord = [0,-32,-10]
  value = combined_atlas.get_fdata()[coord[0],coord[1],coord[2]]
  print(value)
  print(atlas['labels'][int(value)])

  plotting.plot_roi(atlas['maps'],figure=figure,cut_coords=(0,-32,-10))
  plt.show()
    # Create a binary mask of the ROI based on the desired value
  roi_mask = image.math_img('(img == 49)', img=combined_atlas)

  # Plot the ROI
  plotting.plot_roi(roi_mask, title='ROI: Value 49')
  # plotting.plot_roi(ho_maxprob_atlas_img)

  #pprint(atlas['labels'])



  