import concurrent.futures
import librosa
import pandas as pd
import numpy as np
import scipy
import os
from tqdm import tqdm


def columns() -> np.ndarray:
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         mfcc=12, rms=1, spectral_centroid=1, spectral_bandwidth=1, 
                         spectral_contrast=7, spectral_flatness=1, spectral_rolloff=1,
                         poly_features=3, tonnetz=6, zcr=1, dtempo=1,
                         onset_strength=1, tempogram_ratio=13, plp=1)
    single_features = ['onset_num', 'beats', 'tempo', 'dtempo_changes']
    moments = ('mean', 'std', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = (f"{name}_{i:02d}_{moment}" for i in range(size))
            columns.extend(it)
    # columns.extend(single_features)
    columns = np.sort(np.array(columns))
    columns = np.append(columns, single_features)
    columns = np.append(columns, 'Genre')
    return columns


def count_value_changes(arr:list) -> int:
    changes = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            changes += 1
    return changes


def calculate_features_for_single_record(file_path:str) -> list:
    y, sr = librosa.load(file_path)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    rms = librosa.feature.rms(y=y)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    poly_features = librosa.feature.poly_features(y=y, sr=sr, order=2)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    tempogram_ratio = librosa.feature.tempogram_ratio(tg=librosa.feature.tempogram(y=y, sr=sr), sr=sr)
    
    dtempo_changes = count_value_changes(dtempo)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_num = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))

    moments = ['mean', 'std', 'median', 'min', 'max']

    def aggregate_feature(feature):
        return [np.max(feature), np.mean(feature), np.median(feature), np.min(feature), np.std(feature)]
    
    features = []

    for f in [chroma_cens, chroma_cqt, chroma_stft, dtempo, mfcc, onset_env, plp, poly_features, rms, spectral_bandwidth,
              spectral_centroid, spectral_contrast, spectral_flatness, spectral_rolloff, tempogram_ratio, 
              tonnetz, zcr]:
        if f.ndim == 1:
            features.extend(aggregate_feature(f))
        else:
            features.extend(np.hstack([aggregate_feature(f[i]) for i in range(f.shape[0])]))

    features.append(onset_num)
    features.append(len(beats))
    features.append(tempo[0])
    features.append(dtempo_changes)

    genre = file_path.split('/')[-1].split("\\")[0]
    features.append(genre)

    return features


def process_file(file_path:str)->list:
    try:
        features = calculate_features_for_single_record(file_path)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}.")
        return None


def process_files_in_parallel(rootdir:str) -> list(list):
    all_features = []

    # Count total files for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(rootdir) if any(f.endswith('.mp3') for f in files)])

    # Using ThreadPoolExecutor to process files in parallel
    with tqdm(total=total_files, desc="Processing files") as pbar, concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for subdir, dirs, _ in os.walk(rootdir):
            for folder in dirs:
                folder_path = os.path.join(subdir, folder)
                for _, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.mp3'):
                            path = os.path.join(folder_path, file)
                            futures.append(executor.submit(process_file, path))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                all_features.append(result)
            pbar.update(1)

    return all_features



if __name__ == "__main__": 
    # Define root directory and process files
    rootdir = '../datasets/fma/fma_small/'
    df_features = pd.DataFrame(process_files_in_parallel(rootdir), columns=columns())

    # Save the features to CSV
    df_features.to_csv('extracted_features.csv', index=False)