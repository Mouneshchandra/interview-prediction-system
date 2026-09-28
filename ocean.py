import pandas as pd
import numpy as np
from pyDecision.algorithm import fuzzy_ahp_method as fahp

# ---------------------------
# Load OpenFace CSV
# ---------------------------

# ---------------------------
# Helper functions
# ---------------------------
def categorize(value, low=2, high=3):
    """Map continuous AU/head values to Low=1, Medium=2, High=3"""
    if value <= low:
        return 1
    elif value <= high:
        return 2
    else:
        return 3

def classify_score(score):
    """Map normalized score to discrete level 1,2,3"""
    if score <= 1/3:
        return 1
    elif score <= 2/3:
        return 2
    else:
        return 3

def compute_trait(df,features, comparison_matrix, trait_name):
    fuzzy_weights, defuzzified_weights, normalized_weights, rc = fahp(comparison_matrix)
    df[trait_name + "_score"] = 0
    for feature, weight in zip(features, normalized_weights):
        df[trait_name + "_score"] += df[feature] * weight
    min_score = np.sum(np.array(normalized_weights) * 1)
    max_score = np.sum(np.array(normalized_weights) * 3)
    df["norm_" + trait_name + "_score"] = (df[trait_name + "_score"] - min_score) / (max_score - min_score)
    df[trait_name + "_level"] = df["norm_" + trait_name + "_score"].apply(classify_score)
    df[trait_name + "_signal"] = 1 + 2 * df["norm_" + trait_name + "_score"].clip(0, 1)
    print(f"\n{trait_name.capitalize()} Levels:")
    print(df[[trait_name + "_score", "norm_" + trait_name + "_score", trait_name + "_level"]].head())

def ocean_to_hire(path=r"C:\Users\mmm\Downloads\OpenFace_2.2.0_win_x64\output\update_frames.csv", Segment_ID=None):

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Segment_ID"] = Segment_ID
    # ---------------------------
    # 1. Categorize relevant AUs & head movements
    # ---------------------------

    # Openness
    df["AU01_level"] = df["AU01_r"].apply(categorize)
    df["AU02_level"] = df["AU02_r"].apply(categorize)
    df["AU05_level"] = df["AU05_r"].apply(categorize)
    df["head_nod_level"] = df["pose_Rx"].apply(lambda x: categorize(abs(x), low=0.05, high=0.15))
    df["head_tilt_level"] = df["pose_Rz"].apply(lambda x: categorize(abs(x), low=0.05, high=0.15))

    # Conscientiousness
    df["AU23_level"] = df["AU23_r"].apply(categorize)
    df["AU14_level"] = df["AU14_r"].apply(categorize)
    df["AU07_level"] = df["AU07_r"].apply(categorize)
    df["head_still_level"] = df["pose_Ry"].apply(lambda x: categorize(abs(x), low=0.05, high=0.15))

    # Extraversion
    df["AU12_level"] = df["AU12_r"].apply(categorize)
    df["AU06_level"] = df["AU06_r"].apply(categorize)

    # Agreeableness
    df["AU04_level"] = df["AU04_r"].apply(categorize)

    # Neuroticism
    df["AU15_level"] = df["AU15_r"].apply(categorize)
    df["AU14_level"] = df["AU14_r"].apply(categorize)

    # ---------------------------
    # 2. OCEAN trait computations
    # ---------------------------

    # Openness
    openness_matrix = [
        [(1,1,1), (1/3,1/2,1), (1/5,1/4,1/3), (1/7,1/6,1/5)],
        [(3,2,1), (1,1,1), (1/3,1/2,1), (1/5,1/4,1/3)],
        [(5,4,3), (3,2,1), (1,1,1), (1/3,1/2,1)],
        [(7,6,5), (5,4,3), (3,2,1), (1,1,1)]
    ]
    compute_trait(df,["AU01_level","AU02_level","AU05_level","head_nod_level"], openness_matrix, "openness")

    # Conscientiousness
    consc_matrix = [
        [(1,1,1), (1/2, 1/2, 1), (1/3, 1/2, 1), (1/4, 1/3, 1/2)],
        [(2,2,1), (1,1,1), (1/2, 1/2, 1), (1/3, 1/2,1)],
        [(3,2,1), (2,2,1), (1,1,1), (1/2,1/2,1)],
        [(4,3,2), (3,2,1), (2,2,1), (1,1,1)]
    ]
    compute_trait(df,["AU23_level","AU14_level","AU07_level","head_still_level"], consc_matrix, "conscientiousness")

    # Extraversion
    extra_matrix = [
        [(1,1,1), (1/2,1/2,1), (1/4,1/3,1/2)],
        [(2,2,1), (1,1,1), (1/3,1/2,1)],
        [(4,3,2), (3,2,1), (1,1,1)]
    ]
    compute_trait(df,["AU12_level","AU06_level","head_nod_level"], extra_matrix, "extraversion")

    # Agreeableness
    agree_matrix = [
        [(1,1,1), (1/2,1/2,1), (1/4,1/3,1/2)],
        [(2,2,1), (1,1,1), (1/3,1/2,1)],
        [(4,3,2), (3,2,1), (1,1,1)]
    ]
    compute_trait(df,["AU12_level","AU04_level","head_tilt_level"], agree_matrix, "agreeableness")

    # Neuroticism
    neuro_matrix = [
        [(1,1,1), (1/2,1/2,1), (1/4,1/3,1/2)],
        [(2,2,1), (1,1,1), (1/3,1/2,1)],
        [(4,3,2), (3,2,1), (1,1,1)]
    ]
    compute_trait(df,["AU04_level","AU15_level","AU14_level"], neuro_matrix, "neuroticism")
    print(df.columns)
    signal_columns = [
        "openness_signal", "conscientiousness_signal", "extraversion_signal",
        "agreeableness_signal", "neuroticism_signal",
    ]
    mean_signal = df[signal_columns].mean(axis=1)
    df["hire_likert"] = np.clip(1 + 3 * (mean_signal - 1), 1, 7)
    print(df[["hire_likert"]])
    df.to_csv("ocean_hire_output_rt.csv", index=False)
    return df