""" Contains functions related to the manual classification of the pain responses. """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

def basic_classify_subacute_and_chronic(responses_df) -> pd.DataFrame:
  """ 
  Returns a copy of the input pandas dataframe with a new `basicClassification` column, containing the basic classification of each subject
  into `SBPr`, `SBPp` or `chronic` if the group was chronic.

  Takes a pandas dataframe with columns `subject='sub-{i}'` `visit = 'visit{i}'` `group` `average_pain` (although might have other columns too.)
  
  Looks for the last and first session out of the 4 (not including 5) that they were at, and then
  it takes the difference between the pain ratings at those sessions. This classification might fail if the 
  subject only attended one session (but this was only the case for chronic patients) !!
  """
  responses_df['basicClassification'] = '' # create an empty classification

  recovering = []
  persisting = []
  chronic = []

  #remove visit 5
  visits_df = responses_df[responses_df['visit'] != 'visit5']

  for subject in sorted(set(list(visits_df['subject']))):
    visit_df = visits_df.loc[visits_df['subject']==subject]['average_pain']

    if responses_df.loc[responses_df['subject']==subject]['group'].iloc[0] == 'subacute':
      pain_decrease = visit_df.iloc[-1]/visit_df.iloc[0]
      classification = recovering if (pain_decrease < 0.8) else persisting
      classification.append(subject)
      classtype = 'SBPr' if (pain_decrease < 0.8) else 'SBPp'
    else:
      chronic.append(subject)
      classtype ='chronic' 

    responses_df.loc[responses_df['subject'] == subject, 'basicClassification'] = classtype

  print(len(persisting),"persisting subjects")
  print(len(recovering), "recovering subjects")

  groupAverage = np.zeros((4,3))
  for visit_num in [1,2,3,4]:
    # get all subjects that had that visit
    subjects_present = responses_df.loc[responses_df['visit']==f'visit{visit_num}']
    # now average across the three groups
    groups = [recovering, persisting, chronic]
    groupAverage[visit_num-1] = [np.average(subjects_present.loc[subjects_present['subject'].isin(group)]['average_pain']) for group in groups]

  plt.plot(groupAverage)
  plt.title("The pain decrease over 4 runs, (trial 4 - trial 1) averaged \n over all SBPp and SBPr Patients",fontsize=15)
  plt.legend(['recovering','persisting','chronic'],fontsize=15)
  plt.xlabel("Run",fontsize=15)
  plt.ylabel("Average Pain Rating",fontsize=15)
  plt.xticks(np.arange(4), [1,2,3,4])
  plt.show()

  return responses_df


def compute_classifications_for_all_subjects(responses_df) -> pd.DataFrame:
  """
  Returns a copy of the input pandas dataframe with a new `classification` column, containing the linear classification of each subject
  into `SBPr`, `SBPp` or `chronic` if the group was chronic.

  Takes a pandas dataframe with columns `subject='sub-{i}'` `visit = 'visit{i}'` `group` `average_pain` (although might have other columns too.)
  """
  responses_df['classification'] = '' # create an empty classification
  ms = []
  cs = []
  #print(responses_df)
  for subject in sorted(set(list(responses_df['subject']))):
    if responses_df.loc[responses_df['subject']==subject]['group'].iloc[0] == 'subacute':
      visits_df = responses_df.loc[responses_df['subject'] == subject]
      visits_df = visits_df[visits_df['visit'] != 'visit5']
      x = np.array([int(visit[-1]) for visit in visits_df['visit']])

      #now get the best linear fit to that 
      A = np.vstack([x, np.ones(len(x))]).T
      y = visits_df['average_pain']
      m, c = lstsq(A, y, rcond=None)[0]
      ms.append(m)
      cs.append(c)
      if (4*m + c)/(m+c) < 0.8:
        #print((4*m + c)/(m+c))
        classification = 'SBPr'
        #plt.plot(x,visits_df['average_pain'])
        plt.plot(x, m*x + c, label='recovering')
      else:
        classification = 'SBPp'
      
    else:
      classification = 'chronic'
    responses_df.loc[responses_df['subject'] == subject, 'classification'] = classification


  plt.show()

  plt.hist(ms)
  plt.title("The distribution in pain change gradient across all participants")
  plt.show()

  plt.hist(cs)
  plt.title("The distribution in pain change y intercept across all participants")

  plt.show()
  return responses_df


def plot_averages_across_visits(responses_df, classType = 'classification',subjectsToRemove=[],title=""):
  """
  Takes a pandas dataframe with columns `subject`, `classification` or `basicClassification`, `visit`, `average pain`.

  Returns nothing. Plots the average across visits for each group SBPp, SBPr and chronic, with error bars.
  """
  
  responses_df = responses_df.drop(index=responses_df.loc[responses_df['subject'].isin(subjectsToRemove)].index)
  persistingAverages = []
  recoveringAverages = []
  chronicAverages = []
  persistingVars = []
  recoveringVars = []
  chronicVars = []

  # need to remove the mean value of everyone's response before this
  vals = [1,2,3,4]

  for i in vals:
    persistingAverages.append(np.average(responses_df[(responses_df[classType] == 'SBPp') & (responses_df['visit'] == f'visit{i}')]['average_pain']))
    recoveringAverages.append(np.average(responses_df[(responses_df[classType] == 'SBPr') & (responses_df['visit'] == f'visit{i}')]['average_pain']))
    chronicAverages.append(np.average(responses_df[(responses_df[classType] == 'chronic') & (responses_df['visit'] == f'visit{i}')]['average_pain']))
    persistingVars.append(np.std(responses_df[(responses_df[classType] == 'SBPp') & (responses_df['visit'] == f'visit{i}')]['average_pain'],axis=0))
    recoveringVars.append(np.std(responses_df[(responses_df[classType] == 'SBPr') & (responses_df['visit'] == f'visit{i}')]['average_pain'],axis=0))
    chronicVars.append(np.std(responses_df[(responses_df[classType] == 'chronic') & (responses_df['visit'] == f'visit{i}')]['average_pain'],axis=0))
    # print(np.max(responses_df[(responses_df[classType] == 'chronic') & (responses_df['visit'] == f'visit{i}')]['average_pain']))
    # print(np.min(responses_df[(responses_df[classType] == 'chronic') & (responses_df['visit'] == f'visit{i}')]['average_pain']))

  fig = plt.figure(dpi=500)
  plt.errorbar(vals, persistingAverages,label='persisting', yerr=persistingVars,capsize=4)
  plt.errorbar(vals, recoveringAverages, label='recovering', yerr=recoveringVars,capsize=4)
  plt.errorbar(vals, chronicAverages,label='chronic', yerr=chronicVars,capsize=4)
  # plt.plot(vals, persistingAverages,label='persisting')
  # plt.plot(vals, recoveringAverages, label='recovering')
  # plt.plot(vals, chronicAverages,label='chronic')
  plt.legend()
  plt.xlabel('Visit',fontsize=12)
  plt.xticks(vals,vals)
  plt.ylabel("Pain rating",fontsize=12)
  
  classification = "linear gradient" if classType == "classification" else "Last - First visit"
  if title =="":
    plt.title(f"Average pain rating for each group across visits, \n classification = {classification}")
  else:
    plt.title(title, fontsize=15)
  plt.show()


def plot_the_changing_subjects(responses_df,subjectsToPlot, title=""):
  """
  Creates plots of the average pain across all four visits of the subjects listed.

  Returns nothing.
  """
  for subject in subjectsToPlot:
    visits_df = responses_df.loc[responses_df['subject'] == subject]
    visits_df = visits_df[visits_df['visit'] != 'visit5']
    x = np.array([int(visit[-1]) for visit in visits_df['visit']])

    plt.plot(x,visits_df['average_pain'],label=subject)
    plt.title(title)
    plt.xlabel("Visit")
    plt.legend()
    plt.ylabel("Average Pain")

    #now add the best linear fit to that 
    # A = np.vstack([x, np.ones(len(x))]).T
    # y = visits_df['average_pain']
    # m, c = lstsq(A, y, rcond=None)[0]
    #plt.plot(x, m*x + c, label='recovering')