from datasets import Dataset
import pandas as pd

def preprocess_data(data_path):
    """
    Preprocesses a csv containing transcription and reference text columns.
    Parameters:
    - df (pd.DataFrame): A DataFrame with two columns, "transcription" and "reference,"
      each containing sentences to be processed.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """

    df = pd.read_csv(data_path)
    df["transcription"] = df["transcription"].str.lower()
    df["reference"] = df["reference"].str.lower()

    df = df[df["transcription"].str.split().apply(len) >= 3]
    df = df[df["reference"].str.split().apply(len) >= 3]

    df = df[(df["transcription"] != "") & (df["reference"] != "")]

    return df



def save_dataset_to_hf(df, dataset_name, repo_id=None):
    """
    Saves a Pandas DataFrame to Hugging Face Hub as a Dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save.
        dataset_name (str): Dataset name for Hugging Face.
        repo_id (str): Optional Hugging Face repo ID (e.g., "username/repo_name").
    """

    hf_dataset = Dataset.from_pandas(df)
    
    if repo_id:
        hf_dataset.push_to_hub(repo_id)
    else:
        hf_dataset.push_to_hub(dataset_name)
