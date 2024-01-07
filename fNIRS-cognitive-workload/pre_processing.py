import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

def upload_all_subjects(fnirs_data, scaled=False):
    scaler = StandardScaler()
    all_data_dict = {}
    for sub in os.listdir(fnirs_data):
        ## Upload data ##
        sub_ID = sub.split('.csv')[0]
        sub_path = os.path.join(fnirs_data, sub)
        sub_data = pd.read_csv(sub_path)
        
        if scaled == True:
            ## Scale the input data within-subject ##
            X_scaled = scaler.fit_transform(sub_data.drop('label',axis=1).to_numpy())
            scaled_data = pd.DataFrame(X_scaled, columns=sub_data.drop('label',axis=1).columns, index=sub_data.index)
            scaled_data['label'] = sub_data['label']

            ## Add data to dictionary ##
            all_data_dict[sub_ID] = scaled_data
        else:
            all_data_dict[sub_ID] = sub_data
    all_data = pd.concat(all_data_dict)[['AB_I_O','AB_PHI_O','CD_I_O','CD_PHI_O','AB_I_DO','AB_PHI_DO','CD_I_DO','CD_PHI_DO','label']]
    all_data = all_data.rename({'label':'difficulty'}, axis=1)
    
    return all_data


def pca_reduce_data(explained_variance, all_data):
    pca = PCA(n_components=explained_variance, random_state=42)
    X_reduced = pca.fit_transform(all_data.drop('difficulty', axis=1))
    print(f'Number of components remaining required to explain {explained_variance * 100}% of the variance is {pca.n_components_}')

    all_data_reduced = pd.DataFrame(X_reduced,
                                    columns=[f'component_{i + 1}' for i in range(X_reduced.shape[1])],
                                    index=all_data.index
                                    )
    all_data_reduced['difficulty'] = all_data.difficulty

    return X_reduced, all_data_reduced, pca


def detect_block_changes(all_data, label):
    # Create a boolean mask indicating when the label changes (indicating a different task block)
    all_block_change_idx = all_data[label] != all_data[label].shift(1)

    # Use cumsum to create a unique identifier for each task
    all_task_id = all_block_change_idx.cumsum()

    # Group by the unique identifier and create a dictionary of DataFrames
    all_grouped_dataframes = {group: group_df for group, group_df in all_data.groupby(all_task_id)}
    
    return all_grouped_dataframes, all_block_change_idx

def split_chunks(data, t):
    nsamples = t*5.2
    nchunks = data.shape[0] / nsamples
    max_rows_per_df = int(data.shape[0] // nchunks)
    split_array = np.array_split(data, nchunks)
    split_df = pd.concat({i+1:df[:max_rows_per_df] for i, (df) in enumerate(split_array)}).reset_index().set_index(['level_1','level_2'])

    return split_df

def resample_chunks(data, sample_length, n_samples):
    sample_length_frames = math.floor(sample_length*5.2)
    np.random.seed(42)

    # Calculate the maximum starting index for continuous samples
    max_start_index = len(data) - sample_length_frames

    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start_index + 1, size=(n_samples,))

    # Use the starting indices to extract consecutive continuous samples
    sampled_dataframes = {i:data.iloc[start:start + sample_length_frames] for i, (start) in enumerate(start_indices)}
    resampled_df = pd.concat(sampled_dataframes).reset_index().set_index(['level_1','level_2'])
    resampled_df = resampled_df.rename({'level_0':'chunk'}, axis=1, level=0)

    return resampled_df

def sliding_window(df, window_seconds, overlap_seconds):
    if overlap_seconds == 0:
        overlapped_windows = split_chunks(df, window_seconds)
        overlapped_windows = overlapped_windows.reset_index().set_index(['level_1','level_2','level_3'])
        overlapped_windows = overlapped_windows.rename({'level_0':'chunk'}, axis=1)
        overlapped_windows.index.names = ['block','subject','sample']
    else:
        window_size = math.floor(window_seconds*5.2)
        overlap = math.floor(overlap_seconds*5.2)
        windows = []

        # Create rolling window with specified overlap
        for i in range(0, len(df) - window_size + 1, 1):  # Change the step to 1
            window = df.iloc[i:i + window_size]
            windows.append(window)

        # Adjust windows to include overlapping elements
        overlapped_windows = pd.concat({i:df.iloc[i:i + window_size] for i, window in enumerate(range(0, len(df) - window_size + 1, overlap-1))})
        overlapped_windows = overlapped_windows.reset_index().set_index(['level_1','level_2','level_3'])
        overlapped_windows = overlapped_windows.rename({'level_0':'chunk'}, axis=1)
        overlapped_windows.index.names = ['block','subject','sample']

    return overlapped_windows


def partition_subject_indexes(test_size, n_subjects, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Calculate the number of repeats based on sample size
    full_samples = (n_subjects // test_size)
    num_repeats = full_samples + 1 if n_subjects % test_size != 0 else full_samples

    # Create an array of all indexes
    all_indexes = np.arange(n_subjects)

    # Shuffle the array to randomize the order
    np.random.shuffle(all_indexes)

    test_buckets = []
    # Generate sets of random indexes each without replacement
    for i in range(num_repeats):
        start_index = i * test_size
        end_index = (i + 1) * test_size
        random_indexes = all_indexes[start_index:end_index]
        if len(random_indexes) < test_size:
            other_indexes = [x for x in all_indexes if x not in random_indexes]
            resampled_indexes = np.random.choice(other_indexes, test_size - len(random_indexes))
            random_indexes = np.hstack([random_indexes, resampled_indexes])
        test_buckets.append(random_indexes)

    # Check to see if sampling approach worked properly
    print(f'{len(test_buckets)} buckets')
    unique_elements, counts = np.unique(np.array(test_buckets).flatten(), return_counts=True)
    print(f'{len(unique_elements)} unique indexes')
    print('Repeated indexes:', np.where(counts == 2))

    return test_buckets