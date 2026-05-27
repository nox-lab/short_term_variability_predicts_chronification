import matplotlib.pyplot as plt
import pandas as pd

def benjamini_hochberg(df, model = "GBM", prediction="all_visits", alpha=0.05):
    ''' 
    Function that computes the Benjamini-Hochberg (BH) threshold correction
    for false discovery rate limitation

    Params:
    df: dataframe with non-demeaned ratings of subjects and visits, 
    which contains the permutation p-value

    Returns:
    significant_parcels: dataframe containing significant parcels after BH
    FDR correction.
    plot of significance parcels (ROIs) after FDR correction vs their accuracy
    '''

    #Initialising dataframe for output 
    significant_parcels = pd.DataFrame()

    # Sort df by ascending p_value
    sorted_df = df.sort_values(by="p_value").reset_index(drop=True)
    
    #Make rank start at 1
    sorted_df["rank"] = sorted_df.index + 1  

    #Get total number of parcels tested
    m = len(sorted_df)

    # Compute BH threshold per parcel
    sorted_df["BH_threshold"] = (sorted_df["rank"] / m) * alpha

    # Find the largest p-value that <= its own BH threshold
    proportion_FP = sorted_df["p_value"] <= sorted_df["BH_threshold"]
    if proportion_FP.any():
        max_p_value = proportion_FP[proportion_FP].index.max()
        significant_parcels = sorted_df.loc[:max_p_value]
    else:
        significant_parcels = pd.DataFrame(columns=["brain_region", "p_value", "mean_over_folds", "BH_threshold"])

    significant_parcels = significant_parcels.sort_values(by="mean_over_folds", ascending=False)
    
    # Plot top ROIs
    plt.figure(figsize=(12, 6))
    plt.barh(significant_parcels["brain_region"], significant_parcels["mean_over_folds"], color='skyblue')
    plt.xlabel("Mean Accuracy (ROC AUC Score)")
    plt.title("Top ROIs by Accuracy")
    plt.xlim(0.5, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    #plt.savefig(os.path.join(base_path, "Figures", f"{model}_bh_{prediction}_significant.png"))
    plt.show()

    return significant_parcels


def storey_q_value(df, model= "GBM", prediction = "all_visits", lambda_threshold=0.1, alpha=0.05):
    ''' 
    Function that computes storey's q value FDR correction
    
    Params:
    df: dataframe with non-demeaned ratings of subjects and visits, 
    which contains the permutation p-value

    Returns:
    significant_parcels: dataframe containing significant parcels after BH
    FDR correction.
    plot of significance parcels (ROIs) after FDR correction vs their accuracy
    '''

    #Initilaising copy of the df for manipulation
    df = df.copy()

    #Total number of parcels tested
    m = len(df)
    
    # Estimate pi_0 (proportion of TN values)
    proportion_nulls = sum(df["p_value"] > lambda_threshold)
    pi_0 = proportion_nulls / ((1 - lambda_threshold) * m)

    #pi_0 should not exceed 1 (probability)
    pi_0 = min(pi_0, 1.0)  

    # Sort by p-value 
    sorted_df = df.sort_values(by="p_value").reset_index(drop=True)

    #Make rank start from 1
    sorted_df["rank"] = sorted_df.index + 1

    # Calculate q-values
    q_values = (pi_0 * sorted_df["p_value"] * m) / sorted_df["rank"]

    # Enforce monotonicity (q[i] = min(q[i], q[i+1]))
    q_values = q_values[::-1].cummin()[::-1]
    sorted_df["q_value"] = q_values

    # Filter significant parcels by q_value
    significant_df = sorted_df[sorted_df["q_value"] < alpha].copy()

    #Standard deviation across folds
    # significant_df["performance_stats"] = significant_df["performance_stats"].apply(ast.literal_eval)
    # significant_df["std_over_folds"] = significant_df["performance_stats"].apply(lambda x: x["std"])

    significant_df = significant_df.sort_values(by="mean_over_folds", ascending=False)
   
    # Plot top 5 ROIs by mean accuracy
    plt.figure(figsize=(12, 6))
    # xerr=significant_df["std_over_folds"]
    plt.barh(significant_df["brain_region"], significant_df["mean_over_folds"], color='skyblue')
    plt.xlabel("Mean Accuracy (ROC AUC Score)")
    plt.title("Top ROIs by Accuracy")
    plt.xlim(0.5, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    #plt.savefig(os.path.join(base_path, "Figures", f"{model}_storey_qvalue_{prediction}_significant.png"))
    plt.show()

    return significant_df 

