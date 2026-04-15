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
import matplotlib.image as mpimg

# ==============================
# DATA LOADING & PREPROCESSING
# ==============================
def load_training_dataset():
    if not os.path.exists('training_dataset.csv'):
        raise FileNotFoundError(
            "\n❌ training_dataset.csv not found!\n"
            "Please run: python prepare_dataset.py first"
        )
    print("Loading training dataset from: training_dataset.csv")
    return pd.read_csv('training_dataset.csv')


def prepare_features_and_target(df):
    metadata_cols = ['pitcher_id', 'hitter_id', 'encounter_num', 'target_events', 'pa_num', 'bf_num']
    if 'target_events' not in df.columns:
        raise ValueError("target_events column not found in dataset!")
    y = df['target_events'].copy()
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X = df[feature_cols]
    print(f"Extracted {len(feature_cols)} features")
    return X, y


def split_by_encounter(df, val_size=0.15, test_size=0.15, random_state=42):
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
        print("⚠️ Could not find pitcher/hitter ID columns. Falling back to random split.")
        train_idx, temp_idx = train_test_split(range(len(df)), test_size=val_size + test_size, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (val_size + test_size), random_state=random_state)
        return train_idx, val_idx, test_idx

    df = df.copy()
    df['encounter_id'] = df[pitcher_col].astype(str) + '_vs_' + df[hitter_col].astype(str)
    encounters = df['encounter_id'].unique()
    n_encounters = len(encounters)
    print(f"Found {n_encounters} unique encounters")

    np.random.seed(random_state)
    shuffled_encounters = np.random.permutation(encounters)
    n_test  = int(n_encounters * test_size)
    n_val   = int(n_encounters * val_size)
    n_train = n_encounters - n_val - n_test

    train_encounters = set(shuffled_encounters[:n_train])
    val_encounters   = set(shuffled_encounters[n_train:n_train+n_val])
    test_encounters  = set(shuffled_encounters[n_train+n_val:])

    train_idx = df.index[df['encounter_id'].isin(train_encounters)].tolist()
    val_idx   = df.index[df['encounter_id'].isin(val_encounters)].tolist()
    test_idx  = df.index[df['encounter_id'].isin(test_encounters)].tolist()

    total = len(df)
    print(f"\n✅ Split complete:")
    print(f"   Training samples:   {len(train_idx)} ({len(train_idx)/total*100:.1f}%)")
    print(f"   Validation samples: {len(val_idx)} ({len(val_idx)/total*100:.1f}%)")
    print(f"   Testing samples:    {len(test_idx)} ({len(test_idx)/total*100:.1f}%)")
    print("-" * 60)
    return train_idx, val_idx, test_idx


# ==============================
# CLASS WEIGHTS
# ==============================
def get_class_weights(label_encoder, y_encoded):
    classes = np.unique(y_encoded)
    balanced_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
    return {i: w for i, w in zip(classes, balanced_weights)}


# ==============================
# DRAW NETWORK USING MATPLOTLIB
# ==============================
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def draw_network_perfect_circles(layer_sizes, filename='neural_network_final.png'):
    # Colors matching your reference
    colors = {
        "input": "#90EE90",  # Green
        "hidden": "#87CEEB", # Blue
        "output": "#DDA0DD"  # Pink
    }
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('white')
    ax.axis('off')
    
    # CRITICAL: This ensures circles stay circular regardless of axis scaling
    ax.set_aspect('equal')
    
    n_layers = len(layer_sizes)
    x_spacing = 1.5  # Horizontal distance between layers
    node_size = 0.1  # Diameter of the circle
    v_spacing = 0.02 # Vertical distance between nodes

    all_node_coords = []

    for i, n_nodes in enumerate(layer_sizes):
        layer_coords = []
        # Center layers vertically
        total_h = (n_nodes - 1) * v_spacing
        y_start = -total_h / 2

        # Pick color
        if i == 0: color = colors["input"]
        elif i == n_layers - 1: color = colors["output"]
        else: color = colors["hidden"]

        for j in range(n_nodes):
            x = i * x_spacing
            y = y_start + (j * v_spacing)
            layer_coords.append((x, y))
            
            # Using same value for width and height creates a perfect circle
            circle = Ellipse((x, y), width=node_size, height=node_size, 
                             facecolor=color, edgecolor='black', linewidth=0.8, zorder=3)
            ax.add_patch(circle)
            
        all_node_coords.append(layer_coords)

    # Draw Connections
    for i in range(len(all_node_coords) - 1):
        for start_node in all_node_coords[i]:
            for end_node in all_node_coords[i+1]:
                ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 
                        color='black', linewidth=0.2, alpha=0.3, zorder=1)

    # Rescale view to fit everything
    ax.relim()
    ax.autoscale_view()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Define your architecture
layer_sizes = [69, 100, 60, 35, 6] 
draw_network_perfect_circles(layer_sizes)

# MODEL BUILDING
# ==============================
def build_feedforward_model(input_dim, n_classes, plot_path=None):
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
        layers.Dense(n_classes, activation='softmax')
    ])
    print(f"Model input shape: {model.input_shape}")
    if plot_path is not None:
        layer_sizes = [input_dim, 100, 60, 35, n_classes]
        draw_network(layer_sizes, filename=plot_path)
    return model


# ==============================
# EVALUATION
# ==============================
def evaluate_split(model, X, y_true, label_encoder, split_name="Set"):
    print("\n" + "="*60)
    print(f"EVALUATION — {split_name.upper()}")
    print("="*60)
    loss, acc = model.evaluate(X, y_true, verbose=0)
    print(f"\n  Loss:     {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    unique_classes = np.unique(y_true)
    target_names = [label_encoder.classes_[i] for i in unique_classes]

    print(f"\n{'-'*60}\nCLASSIFICATION REPORT\n{'-'*60}")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names, zero_division=0))

    print(f"{'-'*60}\nCONFUSION MATRIX  (Rows = Actual, Columns = Predicted)\n{'-'*60}")
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in target_names],
                         columns=[f"Pred_{c}" for c in target_names])
    print(cm_df)


# ==============================
# MAIN TRAINING PIPELINE
# ==============================
def main():
    print("="*60)
    print("Baseball Outcome Prediction - Training")
    print("="*60)

    df = load_training_dataset()
    X, y = prepare_features_and_target(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    class_weight_dict = get_class_weights(label_encoder, y_encoded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_idx, val_idx, test_idx = split_by_encounter(df)
    X_train, y_train = X_scaled[train_idx], y_encoded[train_idx]
    X_val, y_val = X_scaled[val_idx], y_encoded[val_idx]
    X_test, y_test = X_scaled[test_idx], y_encoded[test_idx]

    # Build model and draw network
    model = build_feedforward_model(X_scaled.shape[1], n_classes, plot_path='neural_network.png')
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    batch_size = min(512, max(32, len(X_train)//100))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=batch_size, class_weight=class_weight_dict, verbose=1)

    # Evaluate
    evaluate_split(model, X_val, y_val, label_encoder, "Validation Set")
    evaluate_split(model, X_test, y_test, label_encoder, "Test Set")

    # Show network image
    try:
        img = mpimg.imread('neural_network.png')
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not display model image: {e}")

    # Save model and artifacts
    model.save('baseball_model.keras')
    with open('model_artifacts.pkl', 'wb') as f:
        pickle.dump({'label_encoder': label_encoder, 'scaler': scaler, 'feature_names': list(X.columns),
                     'n_features': X.shape[1], 'classes': label_encoder.classes_}, f)
    print("✅ Model and artifacts saved.")


if __name__ == "__main__":
    main() 