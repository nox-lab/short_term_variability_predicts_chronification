import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def demean_ratings(actual_ratings):
    '''
    Function that demeans all timeseries ratings

    Params:
    actual_ratings: dataframe containing non-demeaned ratings
    per subject and visit.

    Returns:
    Saves demeaned ratings in a path "Results/Intermediate_Files/"
    '''
    df = actual_ratings.copy()

    #Subtract the mean from columns 4:-3 (timeseries data)
    df.iloc[:, 4:-3] = df.iloc[:, 4:-3].sub(df["average_pain"], axis=0)
    
    #Check to see if the mean of demeaned values is 0 (or very close to 0 - floating point precision)
    print(df.iloc[:, 4:-3].mean().sum())

    #Only keep run_02, if there is both a run_01 and run_02 for a given subject and visit
    df = df.drop_duplicates(subset=['subject', 'visit'], keep = 'last')

    #Save the demeaned ratings intermediate files (used in Hendrix_05_pain_ratings_data_organization)
    df.to_csv("Results/Intermediate_Files/painrating_demeaned_.csv", index = False)

    return df


def delete_0_ratings(df):
    '''
    Deletes subjects in a given visit if they have reported 0 
    consistently for half of the visit or more

    Params:
    df: dataframe containing the nondemeaned ratings per subject 
    and visit.

    Returns:
    df: dataframe containing the nondemeaned ratings per subject 
    and visit excluding those subjects that have more or equal to 30% 
    ratings  = "0"
    '''
    rows_to_remove = []
    
    for v, visit in enumerate(df["visit"]):
        visit_data = df[df["visit"] == visit]
        
        for subject in visit_data["subject"].unique():
            subject_ratings = visit_data[visit_data["subject"] == subject].iloc[:, 4:-3]  # Extract relevant columns
            
            # Convert to numpy array for efficiency
            ratings_array = subject_ratings.values.flatten().tolist()
            count_0s = sum([x==0 for x in ratings_array])
            
            # Check if the longest non-report is at least 30% of the trial
            if count_0s >= len(ratings_array) * 0.3:
                rows_to_remove.append((subject, visit))

    # Remove only the specific rows for those subject-visit combinations
    df_filtered = df[~df.apply(lambda row: (row["subject"], row["visit"]) in rows_to_remove, axis=1)]

    return df_filtered


def calculate_cv(df):
    '''
    Function that calculates the coefficient of variance for 
    each subject in each visit

    Params:
    df: dataframe containing ratings per subject nad visit.

    Returns:
    dataframe containing the veriability CV
    '''
    
    #Take out timeseries columns for eased analysis
    df = pd.concat([df.iloc[:, :4], df.iloc[:, -3:]], axis=1)
    df["CV"] = (df["ratings_std"]/df["average_pain"]) * 100

    return df


# Calculate measure of variance IQR

def remove_non_numeric(data):
    '''
    Function that removes all data that is not numeric
    for IQR calculation

    Params:
    data: ratings per subject and visit (one row in dataframe)

    Returns:
    data filtered of non-numeric values

    '''
    if not isinstance(data, list) and ("0" not in data):  
        return data
    
def compute_row_iqr(row):
    '''
    Function that calculates IQR measure of variance 
    per participant, in one visit

    Params:
    row: rating for one subject and visit (one row in dataframe)

    Returns:
    Interquartile range (IQR) per row
    '''
    row = remove_non_numeric(row)
    Q1 = np.percentile(row, 25)  
    Q3 = np.percentile(row, 75) 
    return Q3 - Q1  


def remove_outliers_iqr(df, group_col, visit_col, value_col):
    filtered = []
    for (g, v), sub_df in df.groupby([group_col, visit_col]):
        q1 = sub_df[value_col].quantile(0.25)
        q3 = sub_df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered.append(sub_df[(sub_df[value_col] >= lower) & (sub_df[value_col] <= upper)])
    return pd.concat(filtered)


def linear_vis_cv(actual_ratings, customised_colours, outliers = False,
                  savefig=False):
    '''
    Function that visualises the linear relationship between visit and CV

    Params:
    actual_ratings: dataframe containing nondemeaned ratings per subject and
    visit.

    Returns:
    plot of coefficient of variation (CV) per patient and visit against visit.
    Saves figure "Results/"
    '''
    plt.figure(figsize=(5, 4))
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]  

    df = actual_ratings.copy()  
    
    df = df.dropna()
    
    # Ensure 'visit' column is categorical with the correct order of visits
    df["visit"] = pd.Categorical(df["visit"], categories=visit_order, ordered=True)
    
    # Sort data based on visit order
    df = df.sort_values("visit")

    if outliers == True:
        strip_data = df

    else:
        strip_data = remove_outliers_iqr(df, group_col="group", visit_col="visit", value_col="CV")

    sns.lineplot(x="visit", y="CV", hue="group", data=strip_data, marker="o", palette=customised_colours)
    
    ax = plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.title("Mean CV Across Subjects Per Group Over 4 Visits")
    plt.ylabel("Mean CV", fontsize=14)
    plt.xlabel("Months", fontsize = 14) 
    ax.set_xticklabels(["1", "2", "3", "4"])
    plt.grid(False)
    plt.legend([], [], frameon=False) 

    # Map visit names to months
    visit_to_month = {
        "visit1": "onset",
        "visit2": "2",
        "visit3": "7",
        "visit4": "13"
    }

    xticks = ax.get_xticks()
    xtick_labels = [visit_to_month.get(v, v) for v in df["visit"].cat.categories]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(axis='both', labelsize=14)

    if savefig:
        plt.savefig("Results/cv_across_visits.png")

    plt.show()


def linear_vis_mean(ratings, customised_colours, outliers = False,
                    savefig = False):
    '''
    Function that visualises the linear relationship between visit and CV

    Params:
    actual_ratings: dataframe containing nondemeaned ratings per subject and
    visit.

    Returns:
    plot of coefficient of variation (CV) per patient and visit against visit.
    Saves figure "Results/"
    '''
    plt.figure(figsize=(5, 4))
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]  
    group_labels = ["Chronic", "SBPp", "SBPr"]

    df = ratings.copy()  
    
    df = df.dropna()
    
    # Ensure 'visit' column is categorical with the correct order of visits
    df["visit"] = pd.Categorical(df["visit"], categories=visit_order, ordered=True)

    # Sort data based on visit order
    df = df.sort_values("visit")

    sns.lineplot(x="visit", y="average_pain", hue="group", data=df, marker="o", palette=customised_colours)
    
    ax = plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.title("Mean CV Across Subjects Per Group Over 4 Visits")
    plt.ylabel("Mean Pain Ratings (0-100)", fontsize = 14)
    plt.xlabel("Months", fontsize = 14) 
    ax.set_xticklabels(["1", "2", "3", "4"])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), ["Chronic", "SBPp", "SBPr"], loc="lower left")
    plt.grid(False)

    # Map visit names to months
    visit_to_month = {
        "visit1": "onset",
        "visit2": "2",
        "visit3": "7",
        "visit4": "13"
    }

    xticks = ax.get_xticks()
    xtick_labels = [visit_to_month.get(v, v) for v in df["visit"].cat.categories]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.tick_params(axis='both', labelsize=14)

    if savefig:
        plt.savefig("Results/mean_across_visits_linear.png", dpi=300, bbox_inches='tight')
    plt.show()


def boxplot_cv(actual_ratings, customised_colours, outliers=False,
               savefig= False):
        '''
        Function that visualises coefficient of variance (CV) in a boxplot

        Params:
        actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
        across visits and groups

        Returns:
        boxplot of the coefficient of variance (CV) per group and visit. 
        Saves figure in "Results/".
        '''
        plt.figure(figsize=(10, 5))

        ax = plt.gca() 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        visit_order = ["visit1", "visit2", "visit3", "visit4"]  
        groups_order = ["chronic", "SBPp", "SBPr"]
        actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
        actual_ratings["group"] = pd.Categorical(actual_ratings["group"], categories=groups_order, ordered=True)

        if outliers == True:
            strip_data = actual_ratings 

        else:
            strip_data = remove_outliers_iqr(actual_ratings, group_col="group", visit_col="visit", value_col="CV")

        sns.stripplot(
        x="visit", y="CV", hue="group", data=strip_data,
        dodge=True, jitter=True, marker="o", alpha=0.5,
        color="black", ax=ax, zorder=1, linewidth=0.5
        )

        # Box plot of CV per group across visits
        ax = sns.boxplot(x="visit", y="CV", hue="group", data=actual_ratings, showfliers=outliers, palette=customised_colours,
                    zorder = 2, ax=ax)
        
        plt.ylabel("CV", fontsize = 14)
        plt.xlabel("Months", fontsize = 14, labelpad=-10)
        # plt.ylim(-1, 69)
        # ax.set_xticklabels(["1", "2", "3", "4"])
        # plt.grid(axis='y')
        plt.grid(False)

            # Map visit names to months
        visit_to_month = {
            "visit1": "onset",
            "visit2": "2",
            "visit3": "7",
            "visit4": "13"
        }

        xticks = ax.get_xticks()
        xtick_labels = [visit_to_month.get(v, v) for v in strip_data["visit"].cat.categories]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), ["Chronic", "SBPp", "SBPr"], loc="upper center", bbox_to_anchor=(0.5, 0.85))
        ax.tick_params(axis='both', labelsize=14)

        if outliers == True and savefig == True:
            plt.savefig("Results/boxplot_CV.png")
        elif outliers != True and savefig ==True:
            plt.savefig("Results/boxplot_CV_no_outliers.png")

        plt.show()


def avg_rating_per_group(actual_ratings, customised_palette, outliers=False,
                         savefig=False):
    '''
    Function that computes and visualises the distribution of average reported pain rating
    per subject, in each group across the 4 visits.

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    boxplot of the average rating per group and visit. 
    Saves figure in "Results/".
    
    '''
    # Create figure with 3 subplots (one for each group)
    groups = actual_ratings["group"].dropna().unique()
    num_groups = len(groups) 
    fig, axes = plt.subplots(1, num_groups, figsize=(10, 4), sharey=True)
    
    # Define groups
    groups = actual_ratings["group"].unique()
    
    # Define colors for consistency
    # palette = sns.color_palette("pastel", num_groups)
    
    # Define correct visit order (swapping visit3 and visit4)
    visit_order = ["visit1", "visit2", "visit3", "visit4"] 
    groups_order = ["chronic", "SBPp", "SBPr"]

    actual_ratings["group"] = pd.Categorical(actual_ratings["group"], categories=groups_order, ordered=True)

    # Map visit names to months
    visit_to_month = {
        "visit1": "onset",
        "visit2": "2",
        "visit3": "7",
        "visit4": "13"
    }
    
    group_name_mapping = ["Chronic", "SBPp", "SBPr"]

    # Loop through each group and create a boxplot
    for i, group in enumerate(groups_order):
        color = customised_palette[group]   
        group_data = actual_ratings[actual_ratings["group"] == group].copy()  
    
        # Ensure visits are categorized and ordered correctly
        group_data["visit"] = pd.Categorical(group_data["visit"], categories=visit_order, ordered=True)
        group_data = group_data.sort_values("visit")  

        axes[i].set_axisbelow(True)
        # Boxplot
        sns.boxplot(x="visit", y="average_pain", data=group_data, ax=axes[i], showfliers=outliers, color=color, zorder=2)
        # Strip plot for individual points
        sns.stripplot(x="visit", y="average_pain", data=group_data, ax=axes[i], jitter=True, alpha=0.4, color="black", zorder=2)
    
        
        # Titles and labels
        if i==0:
            axes[i].set_ylabel("Mean Pain Ratings (0-100)", fontsize = 14)  
        else:
            axes[i].set_ylabel(" ")  
        axes[i].set_title(f"{group_name_mapping[i]}", fontsize = 14, pad=-10)
        axes[i].set_xlabel(" ")
        #axes[i].set_xticklabels(["1", "2", "3", "4"])
        axes[i].grid(False)
        axes[i].set_ylim(0, 100)  
        axes[i].set_yticks(range(0, 101, 10)) 
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        xticks = axes[i].get_xticks()
        xtick_labels = [visit_to_month.get(v, v) for v in group_data["visit"].cat.categories]
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xtick_labels, fontsize=12)
        axes[i].tick_params(axis='y', labelsize=12)

    fig.supxlabel("Months", fontsize=14, y=-0.0001)

    if savefig:
        plt.savefig("Results/boxplot_avg_painrating.png")


def melting_timeseries_in_column(actual_ratings, melted):
    '''
    Function that turns all timeseries from the different columns in 
    one list under column "all_ratings"

    Params:
    actual_ratings: dataframe of the non-demeaned ratings per subjects 
    and visit.

    Returns:
    actual_ratings: dataframe of the non-demeaned ratings per subjects
    and visit which includes a column "all_ratings" that melts all 
    ratings into this column.
    '''
    melted["all_ratings"] = melted.iloc[:, 4:-3].apply(lambda row: row.tolist(), axis=1)
    melted_subset = melted[["subject", "visit", "all_ratings"]]
    
    # Merging `all_ratings` from `melted` into `actual_ratings` based on `subject` and `visit`
    actual_ratings = actual_ratings.merge(melted_subset, on=["subject", "visit"], how="left")
    actual_ratings = actual_ratings.drop(columns=["all_ratings_y"], errors="ignore")
    actual_ratings = actual_ratings.rename(columns={"all_ratings_x": "timeseries_ratings", "all_ratings_y": "timeseries_ratings"})

    return actual_ratings


def plot_timeseries_per_sub(actual_ratings):
    '''  
    Function that plots the timeseries data per subject in each visit and group
    Different visits and groups are displayed in separate subplots.

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    12 subplots, each plot contains timeseries data per group and visit.
    Saves figure "Results/"
    '''
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)

    visits = actual_ratings["visit"].cat.categories
    groups = actual_ratings["group"].dropna().unique()
    
    fig, axes = plt.subplots(len(groups), len(visits), figsize=(17, 7), sharey=True)
    
    
    for visit_idx, visit in enumerate(visits):
        
        visit_ratings = actual_ratings[actual_ratings["visit"] == visit]
    
        #Exclude visit5 from analysis
        if visit != "visit5":
    
            for group_idx, group in enumerate(groups):
    
                ax = axes[group_idx, visit_idx]
    
                timeseries = visit_ratings[visit_ratings["group"] == group]
    
                for _, row in timeseries.iterrows():
                    
                    ax.plot(range(len(row["all_ratings"])), row["all_ratings"], alpha=0.5, lw=0.8)
                
                if group_idx == 0:
                    ax.set_title(f"{visit}", fontsize=12)  
                if visit_idx == 0:
                    ax.set_ylabel(f"{group}", fontsize=12)  
    
                
    plt.suptitle("Timeseries Pain Ratings per Visit and Group", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.savefig("Results/timeseries_ratings_per_subject.png")
    plt.show()


def boxplot_IQR(actual_ratings, outliers = False, savefig =False):
    '''
    Function that visualises a boxplots of the IQR
    by default it excludes outliers 

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    boxplot of Interquartile Range (IQR).
    Saves figure "Results/"
    '''
    # Plotting box plot of the IQR per group across visits
    visit_order = ["visit1", "visit2", "visit3", "visit4"]  
    
    # Convert visits into categorical data with correct order
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="visit", y="IQR", hue="group", data=actual_ratings, palette="pastel", showfliers = outliers)
    
    # Labels and title
    plt.title("Pain Rating Variability (IQR) Across Visits per Group")
    plt.xlabel(" ")
    plt.ylabel("Interquartile Range (IQR)")
    plt.legend(title="Patient Group", loc="upper right")
    
    
    if savefig:
        plt.savefig("Results/boxplot_IQR.png")
    plt.show()


def std_vs_mean(actual_ratings):
    '''
    Function to visualise the relationship between standard deviation
    per patient and the mean

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    standard deviation of the ratings per subject and visit against the mean of the 
    ratings per subject and visit
    Saves figure "Results/
    '''

    visit_order = ["visit1", "visit2", "visit3", "visit4"]
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    visits = actual_ratings["visit"].dropna().unique()
    groups = actual_ratings["group"].dropna().unique()
    
    # Create subplots
    fig, axes = plt.subplots(len(groups), len(visits), figsize=(17, 7), sharey=True, sharex=True)
    
    for gr_nr, group in enumerate(groups):
        grouping = actual_ratings[actual_ratings["group"] == group]
    
        for v, visit in enumerate(visits):
            timeseries = grouping[grouping["visit"] == visit]
    
            ax = axes[gr_nr, v]
    
            # Scatter plot
            ax.scatter(timeseries["average_pain"], timeseries["ratings_std"], color = "black", alpha=0.5, lw=0.8, label="Data")
    
            # Fit and plot best-fit line
            if len(timeseries) > 1:  # Ensure enough points to fit a line
                x = timeseries["average_pain"]
                y = timeseries["ratings_std"]
    
                # Fit linear regression model
                slope, intercept = np.polyfit(x, y, 1)
                best_fit_line = slope * x + intercept
    
                ax.plot(x, best_fit_line, color="red", linestyle="dashed", lw=2, label="Best Fit Line")
    
            # Titles and labels
            if gr_nr == 0:
                ax.set_title(f"{visit}", fontsize=12)
            if v == 0:
                ax.set_ylabel(f"{group}", fontsize=12)
    
    plt.suptitle("Standard Deviation Against Mean per Patient (Best Fit Line)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

    plt.show()


def var_vs_mean(actual_ratings):
    '''
    Function to visualise the relationship between variance
    per patient and the mean

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    variance of the ratings per subject and visit against the mean of the 
    ratings per subject and visit.
    Saves figure "Results/"
    '''
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    visits = actual_ratings["visit"].dropna().unique()
    groups = actual_ratings["group"].dropna().unique()
    
    # Create subplots
    fig, axes = plt.subplots(len(groups), len(visits), figsize=(17, 7), sharey=True, sharex=True)
    
    for gr_nr, group in enumerate(groups):
        grouping = actual_ratings[actual_ratings["group"] == group]
    
        for v, visit in enumerate(visits):
            timeseries = grouping[grouping["visit"] == visit]
    
            ax = axes[gr_nr, v]
    
            # Scatter plot
            ax.scatter(timeseries["average_pain"], timeseries["ratings_variance"], color = "black", alpha=0.5, lw=0.8, label="Data")
    
            # Fit and plot best-fit line
            if len(timeseries) > 1:  # Ensure enough points to fit a line
                x = timeseries["average_pain"]
                y = timeseries["ratings_variance"]
    
                # Fit linear regression model
                slope, intercept = np.polyfit(x, y, 1)
                best_fit_line = slope * x + intercept
    
                ax.plot(x, best_fit_line, color="red", linestyle="dashed", lw=2, label="Best Fit Line")
    
            # Titles and labels
            if gr_nr == 0:
                ax.set_title(f"{visit}", fontsize=12)
            if v == 0:
                ax.set_ylabel(f"{group}", fontsize=12)
    
    plt.suptitle("Variance Against Mean per Patient (Best Fit Line)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.show()


def cv_vs_mean(actual_ratings, customised_palette, savefig=False):
    '''
    Function that visualises the relationship between the CV per subject
    and the mean

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    Plot of CV value per subject (given a group and visit) against their mean pain rating.
    Saves figure "Results/"
    '''
    actual_ratings["ratings_variance"] = actual_ratings["ratings_std"] ** 2
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    visits = actual_ratings["visit"].cat.categories
    groups = actual_ratings["group"].dropna().unique()
    
    # Create subplots
    fig, axes = plt.subplots(len(groups), len(visits), figsize=(17, 7), sharey=True, sharex=True)

    customised_colours = list(customised_palette.values())
    
    for gr_nr, group in enumerate(groups):

        custom_colour = customised_colours[gr_nr]
        grouping = actual_ratings[actual_ratings["group"] == group]
    
        for v, visit in enumerate(visits):
            timeseries = grouping[grouping["visit"] == visit]
    
            ax = axes[gr_nr, v]

            ax.set_axisbelow(True)
    
            # Scatter plot
            ax.scatter(timeseries["average_pain"], timeseries["CV"], color = custom_colour, alpha=0.5, lw=0.8, label="Data")

            ax.grid(False)
            # Titles and labels
            if gr_nr == 0:
                ax.set_title(f"{visit}", fontsize=11)
            if v == 0:
                ax.set_ylabel(f"{group}", fontsize=11)
    
    fig.supxlabel("Mean Pain Ratings", fontsize=12)
    fig.supylabel("CV", fontsize = 12, x=0.01) 
    plt.suptitle("CV against Mean Pain Ratings per Subject", fontsize=16, fontweight="bold")
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.tight_layout()  # Adjust layout to fit suptitle
    if savefig:
        plt.savefig("Results/cv_vs_mean_with_fit.png", dpi=300, bbox_inches="tight")
    plt.show()


def vis_ratings_with_IQR(actual_ratings, actual_ratings_melted):
    '''
    Function that plots the distribution of ratings in a visit, with IQR overimposed
    
    Params:
    actual_ratings: dataframe that contains the non-demeaned timeseries ratings for each 
    subject in every group and visit.
    actual_ratings_melted: dataframe that features the non-demeaned timeseries ratings for 
    each subject in every group and visit in one column.

    Returns:
    violin plot of data per visit with overimposed IQR per visit.
    Saves figure "Results/"
    '''
    
    visit_order = ["visit1", "visit2", "visit3", "visit4"]
    actual_ratings_melted["visit"] = pd.Categorical(actual_ratings_melted["visit"], categories=visit_order, ordered=True)
    actual_ratings_melted["IQR"] = actual_ratings["IQR"].copy()
    
    # Explode ratings into rows
    exploded = actual_ratings_melted.explode("all_ratings").copy()

    # Convert ratings to numeric
    exploded["all_ratings"] = pd.to_numeric(exploded["all_ratings"], errors="coerce")
    exploded = exploded.dropna(subset=["all_ratings"])
    exploded["IQR"] = actual_ratings["IQR"].copy()

    # Get visit and group lists
    visits = visit_order
    groups = exploded["group"].dropna().unique()
    
    # Setup plot
    fig, axes = plt.subplots(len(groups), len(visits), figsize=(17, 7), sharey=True)

    for gr_idx, group in enumerate(groups):
        for v_idx, visit in enumerate(visits):
            ax = axes[gr_idx, v_idx]
            data = exploded[(exploded["group"] == group) & (exploded["visit"] == visit)]

            
            #Adding information about skewness of the distribution
            group_skew = skew(data["all_ratings"])
            group_kurt = kurtosis(data["all_ratings"], fisher=True)  # Fisher=True → normal distribution has kurtosis 0
            
            print(f"{group} - {visit}: Skewness = {group_skew:.2f}, Kurtosis = {group_kurt:.2f}")

            
            # Violin plot
            sns.violinplot(y=data["all_ratings"], ax=ax, color="lightgray", inner=None, linewidth=0)

            # Draw IQR range as a red box or line
            ax.hlines(data["IQR"], xmin=-0.2, xmax=0.2, color='red', linewidth=3)

            # Titles
            if gr_idx == 0:
                ax.set_title(visit)
            if v_idx == 0:
                ax.set_ylabel(group)
            else:
                ax.set_ylabel("")
            ax.set_xticks([])

    plt.suptitle("Pain Rating Distributions with IQR Overlay", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def boxplot_variance(actual_ratings, outliers=False, savefig=False):
    ''' 
    Function that visualises a boxplots of the variance 
    by default it excludes outliers 

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    boxplot of variance.
    Saves figure "Results/"
    '''
    # Plotting box plot of the IQR per group across visits
    visit_order = ["visit1", "visit2", "visit3", "visit4"]  
    
    # Convert visits into categorical data with correct order
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="visit", y="ratings_variance", hue="group", data=actual_ratings, palette="pastel", showfliers=outliers)
    
    # Labels and title
    plt.title("Variance Across Visits per Group")
    plt.xlabel(" ")
    plt.ylabel("Variance")
    plt.legend(title="Patient Group", loc="upper right")
    
    if savefig:
        plt.savefig("Results/boxplot_variance_no outliers.png")
    plt.show()


def boxplot_std(actual_ratings, outliers=False, savefig =False):
    ''' 
    Function that visualises a boxplots of the standard deviation 
    by default it excludes outliers 

    Params:
    actual_ratings: dataframe containing the subjects' non-demeaned timeseries ratings 
    across visits and groups

    Returns:
    boxplot of standard deviation across groups and visits + boxplot saved in path:
    "Results/"
    '''
    visit_order = ["visit1", "visit2", "visit3", "visit4"]  
    
    # Convert visits into categorical data with correct order
    actual_ratings["visit"] = pd.Categorical(actual_ratings["visit"], categories=visit_order, ordered=True)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="visit", y="ratings_std", hue="group", data=actual_ratings, palette="pastel", showfliers = outliers)
    
    # Labels and title
    plt.title("Standard Deviation Across Visits per Group")
    plt.xlabel(" ")
    plt.ylabel("Std")
    plt.legend(title="Patient Group", loc="upper right")
    
    if savefig:
        plt.savefig("Results/boxplot_std.png")
    plt.show()





