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


def split_by_encounter(df, test_size=0.2, random_state=42):
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
        return train_test_split(
            range(len(df)),
            test_size=test_size,
            random_state=random_state
        )

    print(f"Using columns: {pitcher_col} and {hitter_col}")

    df['encounter_id'] = df[pitcher_col].astype(str) + '_vs_' + df[hitter_col].astype(str)

    encounter_counts = df['encounter_id'].value_counts()
    n_encounters = len(encounter_counts)

    print(f"Found {n_encounters} unique pitcher-hitter encounters")
    print(f"Encounter sizes: min={encounter_counts.min()}, "
          f"max={encounter_counts.max()}, "
          f"mean={encounter_counts.mean():.1f}")

    np.random.seed(random_state)
    train_indices = []
    test_indices = []

    encounters_processed = 0

    for encounter_id, group in df.groupby('encounter_id'):
        group_indices = group.index.tolist()
        n_samples = len(group_indices)

        shuffled_indices = np.random.permutation(group_indices)
        n_train = max(1, int(n_samples * (1 - test_size)))

        train_indices.extend(shuffled_indices[:n_train])
        test_indices.extend(shuffled_indices[n_train:])

        encounters_processed += 1
        if encounters_processed % 100 == 0:
            print(f"  Processed {encounters_processed}/{n_encounters} encounters...")

    df.drop('encounter_id', axis=1, inplace=True)

    print(f"\n✅ Split complete:")
    print(f"   Training samples: {len(train_indices)} ({len(train_indices)/len(df)*100:.1f}%)")
    print(f"   Testing samples:  {len(test_indices)} ({len(test_indices)/len(df)*100:.1f}%)")
    print(f"   Each of {n_encounters} encounters split {int((1-test_size)*100)}/{int(test_size*100)}")
    print("-" * 60)

    return train_indices, test_indices


# ==============================
# HARDCODED CLASS WEIGHTS
# ==============================
def get_class_weights(label_encoder):
    """
    Hardcoded class weights.
    Tuning targets:
      - precision > 0.30 per class
      - recall > 0.30 per class
      - macro avg f1 > 0.45
    """
    raw_weights = {
        'double':    3.0,
        'field_out': 1.0,
        'home_run':  2.5,
        'single':    1.2,
        'triple':    8.0,
        'walk':      0.6
    }

    class_weight_dict = {}
    for i, class_name in enumerate(label_encoder.classes_):
        if class_name not in raw_weights:
            raise ValueError(f"No hardcoded weight found for class: '{class_name}'")
        class_weight_dict[i] = raw_weights[class_name]

    return class_weight_dict, raw_weights


# ==============================
# MODEL BUILDING
# ==============================

def build_feedforward_model(input_dim, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        #layers.Dense(80, activation='relu'),
        #layers.BatchNormalization(),
        #layers.Dropout(0.3),

        #layers.Dense(46, activation='relu'),
        #layers.BatchNormalization(),
        #layers.Dropout(0.3),

        #layers.Dense(27, activation='relu'),
        #layers.BatchNormalization(),
        #layers.Dropout(0.3),

        layers.Dense(90, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(45, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(n_classes, activation='softmax')
    ])

    print(f"Model input shape: {model.input_shape}")
    return model


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

    print("\n[3.5/7] Using hardcoded raw class weights...")
    class_weight_dict, raw_weights = get_class_weights(label_encoder)

    print("\nClass weights:")
    print(f"{'Class':<15} {'Weight':<12} {'Sample Count':<15}")
    print("-" * 45)
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"{class_name:<15} {class_weight_dict[i]:>10.2f}       {count:>10,}")

    print("\n[4/7] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features normalized using StandardScaler")

    print("\n[5/7] Splitting data by encounter...")
    train_indices, test_indices = split_by_encounter(df, test_size=0.2, random_state=42)

    X_train = X_scaled[train_indices]
    y_train = y_encoded[train_indices]

    X_test = X_scaled[test_indices]
    y_test = y_encoded[test_indices]

    print("\nTarget distribution in train set:")
    train_target_names = [label_encoder.classes_[i] for i in y_train]
    print(pd.Series(train_target_names).value_counts())

    print("\nTarget distribution in test set:")
    test_target_names = [label_encoder.classes_[i] for i in y_test]
    print(pd.Series(test_target_names).value_counts())

    print("\n[6/7] Building and training model...")
    model = build_feedforward_model(
        input_dim=X_scaled.shape[1],
        n_classes=n_classes
    )

    # Changed learning rate from 0.1 to 0.001 for stable training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.000001,
        verbose=1
    )

    batch_size = min(512, max(32, len(X_train) // 100))

    print(f"\nTraining with batch size: {batch_size}")
    print("-" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n[7/7] Evaluating model...")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nOverall Test Accuracy: {test_acc:.4f}")
    print(f"Overall Test Loss: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)

    unique_classes_in_test = np.unique(y_test)
    target_names_filtered = [label_encoder.classes_[i] for i in unique_classes_in_test]

    print(classification_report(
        y_test, y_pred,
        labels=unique_classes_in_test,
        target_names=target_names_filtered,
        zero_division=0
    ))

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes_in_test)
    print("Rows = Actual, Columns = Predicted")
    print(f"\nClasses in test set: {target_names_filtered}\n")

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in target_names_filtered],
        columns=[f"Pred_{c}" for c in target_names_filtered]
    )
    print(cm_df)

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