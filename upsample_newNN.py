import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import matplotlib.pyplot as plt


# ==============================
# DATA LOADING & PREPROCESSING
# ==============================
def load_training_dataset():
   """Load the prepared training dataset."""
   if not os.path.exists('training_dataset.csv'):
       raise FileNotFoundError(
           "\n❌ training_dataset.csv not found!\n"
           "Please run: python prepare_dataset.py first"
       )
  
   print("Loading training dataset from: training_dataset.csv")
   df = pd.read_csv('training_dataset.csv')
   return df


def prepare_features_and_target(df):
   """Separate features from target variable."""
   metadata_cols = ['pitcher_id', 'hitter_id', 'encounter_num', 'target_events',
                     'pa_num', 'bf_num']
  
   if 'target_events' not in df.columns:
       raise ValueError("target_events column not found in dataset!")
  
   y = df['target_events'].copy()
  
   feature_cols = [col for col in df.columns if col not in metadata_cols]
   X = df[feature_cols]
  
   print(f"Extracted {len(feature_cols)} features")
  
   return X, y


def split_by_encounter(df, val_size=0.15, test_size=0.15, random_state=42):
   """
   Split data into train/val/test sets by assigning whole encounters to each split.
   This prevents data leakage (no pitcher-hitter pair appears in multiple splits)
   and ensures the correct row distribution (~70/15/15).
   """
   print("\n[ENCOUNTER-LEVEL SPLIT]")
   print("-" * 60)

   pitcher_col = None
   hitter_col = None

   for col in df.columns:
       if 'Pitcher_MLBAMID' in col or 'pitcher_id' in col.lower():
           pitcher_col = col
       if 'Hitter_MLBAMID' in col or 'hitter_id' in col.lower() or 'batter' in col.lower():
           hitter_col = col

   if pitcher_col is None or hitter_col is None:
       print("⚠️  Warning: Could not find pitcher/hitter ID columns!")
       print("   Available columns:", df.columns.tolist()[:10], "...")
       print("   Falling back to random split...")
       train_idx, temp_idx = train_test_split(
           range(len(df)), test_size=val_size + test_size, random_state=random_state
       )
       val_idx, test_idx = train_test_split(
           temp_idx, test_size=test_size / (val_size + test_size), random_state=random_state
       )
       return train_idx, val_idx, test_idx

   print(f"Using columns: {pitcher_col} and {hitter_col}")

   df = df.copy()
   df['encounter_id'] = df[pitcher_col].astype(str) + '_vs_' + df[hitter_col].astype(str)

   encounters = df['encounter_id'].unique()
   n_encounters = len(encounters)
   print(f"Found {n_encounters} unique pitcher-hitter encounters")

   encounter_counts = df['encounter_id'].value_counts()
   print(f"Encounter sizes: min={encounter_counts.min()}, "
         f"max={encounter_counts.max()}, "
         f"mean={encounter_counts.mean():.1f}")

   np.random.seed(random_state)
   shuffled_encounters = np.random.permutation(encounters)

   n_test  = int(n_encounters * test_size)
   n_val   = int(n_encounters * val_size)
   n_train = n_encounters - n_val - n_test

   train_encounters = set(shuffled_encounters[:n_train])
   val_encounters   = set(shuffled_encounters[n_train:n_train + n_val])
   test_encounters  = set(shuffled_encounters[n_train + n_val:])

   train_idx = df.index[df['encounter_id'].isin(train_encounters)].tolist()
   val_idx   = df.index[df['encounter_id'].isin(val_encounters)].tolist()
   test_idx  = df.index[df['encounter_id'].isin(test_encounters)].tolist()

   total = len(df)
   print(f"\n✅ Split complete:")
   print(f"   Training samples:   {len(train_idx)} ({len(train_idx)/total*100:.1f}%)")
   print(f"   Validation samples: {len(val_idx)} ({len(val_idx)/total*100:.1f}%)")
   print(f"   Testing samples:    {len(test_idx)} ({len(test_idx)/total*100:.1f}%)")
   print(f"   Encounters — train: {len(train_encounters)}, val: {len(val_encounters)}, test: {len(test_encounters)}")
   print("-" * 60)

   return train_idx, val_idx, test_idx


# ==============================
# BALANCED CLASS WEIGHTS
# ==============================
def get_class_weights(label_encoder, y_encoded):
   """
   Compute balanced class weights automatically using sklearn.
   Rare classes (e.g. triple, home_run) receive higher weights;
   common classes (e.g. field_out) receive lower weights.
   Formula: weight = n_samples / (n_classes * class_count)
   """
   classes = np.unique(y_encoded)
   balanced_weights = compute_class_weight(
       class_weight='balanced',
       classes=classes,
       y=y_encoded
   )

   class_weight_dict = {i: w for i, w in zip(classes, balanced_weights)}

   return class_weight_dict


# ==============================
# MODEL BUILDING
# ==============================

def build_feedforward_model(input_dim, n_classes):
   model = keras.Sequential([
       layers.Input(shape=(input_dim,)),
     
       layers.Dense(100, activation='relu'),
       layers.BatchNormalization(),
       layers.Dropout(0.3),

       layers.Dense(60, activation='relu'),
       layers.BatchNormalization(),
       layers.Dropout(0.3),
      
       layers.Dense(35, activation='relu'),
       layers.BatchNormalization(),
       layers.Dropout(0.3),

       #layers.Dense(44, activation='relu'),
       #layers.BatchNormalization(),
       #layers.Dropout(0.3),

       #layers.Dense(26, activation='relu'),
       #layers.BatchNormalization(),
       #layers.Dropout(0.2),

       layers.Dense(n_classes, activation='softmax')
   ])
  
   print(f"Model input shape: {model.input_shape}")
   return model


# ==============================
# EVALUATION HELPER
# ==============================
def evaluate_split(model, X, y_true, label_encoder, split_name="Set"):
   """Print loss, accuracy, classification report, and confusion matrix for a split."""
   print("\n" + "=" * 60)
   print(f"EVALUATION — {split_name.upper()}")
   print("=" * 60)

   loss, acc = model.evaluate(X, y_true, verbose=0)
   print(f"\n  Loss:     {loss:.4f}")
   print(f"  Accuracy: {acc:.4f}")

   y_pred_probs = model.predict(X, verbose=0)
   y_pred = np.argmax(y_pred_probs, axis=1)

   unique_classes = np.unique(y_true)
   target_names = [label_encoder.classes_[i] for i in unique_classes]

   print(f"\n{'-'*60}")
   print("CLASSIFICATION REPORT")
   print(f"{'-'*60}")
   print(classification_report(
       y_true, y_pred,
       labels=unique_classes,
       target_names=target_names,
       zero_division=0
   ))

   print(f"{'-'*60}")
   print("CONFUSION MATRIX  (Rows = Actual, Columns = Predicted)")
   print(f"{'-'*60}")
   cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
   cm_df = pd.DataFrame(
       cm,
       index=[f"Actual_{c}" for c in target_names],
       columns=[f"Pred_{c}" for c in target_names]
   )
   print(cm_df)


# ==============================
# MAIN TRAINING PIPELINE
# ==============================
def main():
   print("=" * 60)
   print("Baseball Outcome Prediction - Training")
   print("=" * 60)
  
   print("\n[1/7] Loading training dataset...")
   try:
       df = load_training_dataset()
   except FileNotFoundError as e:
       print(f"\n{e}")
       print("\nWorkflow:")
       print("  1. Run: python lookup.py (collect data)")
       print("  2. Run: python prepare_dataset.py (prepare training data)")
       print("  3. Run: python newNN.py (train model)")
       return
  
   print(f"✅ Loaded {len(df)} training examples")
  
   print("\n[2/7] Preparing features and target...")
   X, y = prepare_features_and_target(df)
  
   print(f"Features shape: {X.shape}")
   print(f"Number of features: {X.shape[1]}")
   print(f"\nTarget distribution:")
   print(y.value_counts())
  
   print("\n[3/7] Encoding target variable...")
   label_encoder = LabelEncoder()
   y_encoded = label_encoder.fit_transform(y)
   n_classes = len(label_encoder.classes_)
   print(f"Number of classes: {n_classes}")
   print(f"Classes: {label_encoder.classes_}")
  
   print("\n[3.5/7] Computing balanced class weights...")
   class_weight_dict = get_class_weights(label_encoder, y_encoded)

   print("\nClass weights (higher = rarer class):")
   print(f"{'Class':<15} {'Weight':<12} {'Sample Count':<15}")
   print("-" * 45)
   for i, class_name in enumerate(label_encoder.classes_):
       count = np.sum(y_encoded == i)
       print(f"{class_name:<15} {class_weight_dict[i]:>10.4f}       {count:>10,}")

   print("\n[4/7] Normalizing features...")
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   print("✅ Features normalized using StandardScaler")
  
   print("\n[5/7] Splitting data by encounter (70 / 15 / 15)...")
   train_indices, val_indices, test_indices = split_by_encounter(
       df, val_size=0.15, test_size=0.15, random_state=42
   )
  
   X_train = X_scaled[train_indices]
   y_train = y_encoded[train_indices]

   X_val = X_scaled[val_indices]
   y_val = y_encoded[val_indices]
  
   X_test = X_scaled[test_indices]
   y_test = y_encoded[test_indices]
  
   for split_name, y_split in [("train", y_train), ("validation", y_val), ("test", y_test)]:
       names = [label_encoder.classes_[i] for i in y_split]
       print(f"\nTarget distribution in {split_name} set:")
       print(pd.Series(names).value_counts())

   print("\n[6/7] Building and training model...")
   model = build_feedforward_model(
       input_dim=X_scaled.shape[1],
       n_classes=n_classes
   )

   model.compile(
       optimizer=keras.optimizers.Adam(learning_rate=0.001),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
  
   print("\nModel Architecture:")
   model.summary()
  
   batch_size = min(512, max(32, len(X_train) // 100))
  
   print(f"\nTraining with batch size: {batch_size}")
   print("-" * 60)
  
   history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=10,
       batch_size=batch_size,
       class_weight=class_weight_dict,
       verbose=1
   )
  
   print("\n[7/7] Evaluating model...")

   evaluate_split(model, X_val, y_val, label_encoder, split_name="Validation Set")
   evaluate_split(model, X_test, y_test, label_encoder, split_name="Test Set")

   print("\n" + "=" * 60)
   print("SAVING MODEL")
   print("=" * 60)
  
   model.save('baseball_model.keras')
   print("✅ Model saved as 'baseball_model.keras'")
  
   with open('model_artifacts.pkl', 'wb') as f:
       pickle.dump({
           'label_encoder': label_encoder,
           'scaler': scaler,
           'feature_names': list(X.columns),
           'n_features': X.shape[1],
           'classes': label_encoder.classes_
       }, f)
   print("✅ Model artifacts saved as 'model_artifacts.pkl'")
  
   print("\n" + "=" * 60)
   print("TRAINING COMPLETE!")
   print("=" * 60)
  
if __name__ == "__main__":
   main()