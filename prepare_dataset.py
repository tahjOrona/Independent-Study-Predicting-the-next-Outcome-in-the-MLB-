import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

"""
This script prepares the training dataset from the combined_training_data.csv file.
Each row in the input already contains: Pitcher stats + Hitter stats + Encounter data.
We just need to clean it up and encode it for training.
"""

def load_combined_data(filepath='combined_training_data.csv'):
    """Load the combined training data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def filter_complete_rows(df):
    """
    Filter to only include rows where we have both pitcher and hitter statistics.
    """
    print("\nFiltering for complete data...")
    
    initial_count = len(df)
    
    # Check if Pitcher_Available and Hitter_Available columns exist
    if 'Pitcher_Available' in df.columns:
        df = df[df['Pitcher_Available'] != 'No']
    
    if 'Hitter_Available' in df.columns:
        df = df[df['Hitter_Available'] != 'No']
    
    # Alternatively, check for key columns
    required_pitcher_col = 'Pitcher_MLBAMID'
    required_hitter_col = 'Hitter_MLBAMID'
    
    if required_pitcher_col in df.columns and required_hitter_col in df.columns:
        df = df[df[required_pitcher_col].notna()]
        df = df[df[required_hitter_col].notna()]
    
    filtered_count = len(df)
    removed = initial_count - filtered_count
    
    print(f"  Kept {filtered_count} rows with complete data")
    print(f"  Removed {removed} rows missing pitcher/hitter stats ({removed/initial_count*100:.1f}%)")
    
    return df

def prepare_target(df):
    """
    Extract and map the target variable (outcome of the encounter).
    """
    print("\nPreparing target variable...")
    
    # The target should be in the Encounter_events column
    if 'Encounter_events' not in df.columns:
        raise ValueError("Encounter_events column not found!")
    
    target_col = 'Encounter_events'
    
    # Remove rows with missing outcomes
    initial_count = len(df)
    df = df[df[target_col].notna()]
    removed = initial_count - len(df)
    print(f"  Removed {removed} rows with missing outcomes")
    
    # Map outcomes to simplified categories
    outcome_mapping = {
        'field_out': 'field_out',
        'grounded_into_double_play': 'field_out',
        'fielders_choice': 'field_out',
        'sac_fly': 'field_out',
        'strikeout': 'field_out',
        'force_out': 'field_out',
        'single': 'single',
        'double': 'double',
        'triple': 'triple',
        'home_run': 'home_run',
        'walk': 'walk',
        'hit_by_pitch': 'walk',
        'intent_walk': 'walk'
    }
    
    df['target_events'] = df[target_col].map(outcome_mapping)
    
    # Remove unmapped outcomes
    before_filter = len(df)
    df = df[df['target_events'].notna()]
    removed = before_filter - len(df)
    print(f"  Removed {removed} rows with unmapped outcomes")
    
    # Show distribution
    print("\n📊 Target distribution:")
    print(df['target_events'].value_counts())
    
    return df

def prepare_features(df):
    """
    Prepare features for training by separating them from metadata.
    """
    print("\nPreparing features...")
    
    # Columns to exclude from features (metadata and target)
    exclude_cols = [
        'target_events',#
        'Encounter_events',  # Original target
        'Encounter_player_name',#
        'Encounter_batter',#
        'Encounter_pitcher',#
        'Encounter_description',
        'Pitcher_NameASCII',
        'Pitcher_PlayerId',
        'Hitter_Name',
        'Hitter_NameASCII',
        'Hitter_PlayerId',
        'Pitcher_Available',
        
    ]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['target_events'].copy()
    
    print(f"  Selected {len(feature_cols)} feature columns")
    
    # Show feature breakdown
    pitcher_features = [c for c in feature_cols if c.startswith('Pitcher_')]
    hitter_features = [c for c in feature_cols if c.startswith('Hitter_')]
    encounter_features = [c for c in feature_cols if c.startswith('Encounter_')]
    
    print(f"\n  Feature breakdown:")
    print(f"    • Pitcher features: {len(pitcher_features)}")
    print(f"    • Hitter features: {len(hitter_features)}")
    print(f"    • Encounter features: {len(encounter_features)}")
    
    return X, y, feature_cols

def encode_categorical_features(X):
    """Encode categorical columns using label encoding."""
    print("\nEncoding categorical features...")
    
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"  Found {len(categorical_cols)} categorical features")
    
    # Create encoded version
    X_encoded = X.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = X_encoded[col].fillna('missing')
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
    
    print(f"  ✅ Encoded {len(encoders)} categorical features")
    
    return X_encoded, encoders

def main():
    print("=" * 70)
    print("Preparing Training Dataset - Encounter-First Approach")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1/6] Loading combined data...")
    df = load_combined_data('combined_training_data.csv')
    
    # Step 2: Filter for complete rows
    print("\n[Step 2/6] Filtering for complete data...")
    df = filter_complete_rows(df)
    
    if len(df) == 0:
        print("\n❌ No complete rows found!")
        print("Check if your player statistics files (FHH.csv, FHP.csv) have the correct MLBAMIDs")
        return
    
    # Step 3: Prepare target
    print("\n[Step 3/6] Preparing target variable...")
    df = prepare_target(df)
    
    if len(df) == 0:
        print("\n❌ No valid outcomes found!")
        return
    
    # Step 4: Prepare features
    print("\n[Step 4/6] Preparing features...")
    X, y, feature_cols = prepare_features(df)
    
    # Step 5: Save original (human-readable) version
    print("\n[Step 5/6] Saving original dataset...")
    original_df = X.copy()
    original_df['target_events'] = y
    original_df.to_csv('training_dataset_original.csv', index=False)
    print("  ✅ Saved: training_dataset_original.csv (human-readable)")
    
    # Step 6: Encode and save ML-ready version
    print("\n[Step 6/6] Encoding for machine learning...")
    X_encoded, encoders = encode_categorical_features(X)
    
    # Fill any remaining NaN with 0
    X_encoded = X_encoded.fillna(0)
    
    # Save encoded version
    encoded_df = X_encoded.copy()
    encoded_df['target_events'] = y
    encoded_df.to_csv('training_dataset.csv', index=False)
    print("  ✅ Saved: training_dataset.csv (ML-ready)")
    
    # Save encoders and metadata
    with open('model_artifacts_prep.pkl', 'wb') as f:
        pickle.dump({
            'encoders': encoders,
            'feature_names': feature_cols,
            'n_features': len(feature_cols)
        }, f)
    print("  ✅ Saved: model_artifacts_prep.pkl")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Final Dataset:")
    print(f"   • Total rows: {len(encoded_df)}")
    print(f"   • Total features: {len(feature_cols)}")
    print(f"   • Target classes: {y.nunique()}")
    
    print(f"\n📁 Files created:")
    print(f"   1. training_dataset_original.csv - Human-readable")
    print(f"   2. training_dataset.csv - ML-ready (use this for training)")
    print(f"   3. model_artifacts_prep.pkl - Encoders and metadata")
    
    print(f"\n✅ Ready for training! Run newNN.py to train the model.")

if __name__ == "__main__":
    main()