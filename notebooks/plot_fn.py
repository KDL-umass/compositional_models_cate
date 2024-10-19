import matplotlib.pyplot as plt
import seaborn as sns


# def plot functions 

# 1. # Plotting the distribution of the query execution time per treatment ID
def plot_output_treatment_id(df, output_column,treatment_ids):
    df_0 = df[df["treatment_id"] == treatment_ids[0]]
    df_1 = df[df["treatment_id"] == treatment_ids[1]]

    # density plot
    sns.kdeplot(df_0[output_column], label="Treatment 0")
    sns.kdeplot(df_1[output_column], label="Treatment 1")
    plt.xlabel("Query Execution Time")
    plt.ylabel("Density")
    plt.title("Distribution of Query Execution Time per Treatment ID")
    plt.legend()
    plt.show()
