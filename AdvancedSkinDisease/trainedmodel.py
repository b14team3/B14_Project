import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from tqdm import tqdm
from joblib import dump  # For saving models

train_dir = "skin-disease-dataset/train_set"
test_dir = "skin-disease-dataset/test_set"

def extract_features(img):
    """Extract multiple features including HOG, color histograms, and texture features"""
    # Resize image
    img = cv2.resize(img, (128, 128))
    
    features = []
    
    # HOG Features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_features = hog.compute(equalized)
    features.extend(hog_features.flatten())

    # Color Histograms (RGB and HSV)
    for color in range(3):
        hist = cv2.calcHist([img], [color], None, [256], [0, 256])
        features.extend(hist.flatten())
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
        features.extend(hist.flatten())

    return np.array(features)

def load_data(directory):
    """Load images and extract features and labels"""
    data = []
    labels = []
    classes = os.listdir(directory)
    
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    features = extract_features(img)
                    if features is not None and len(features) > 0:
                        data.append(features)
                        labels.append(class_name)
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(data), np.array(labels)

def create_voting_ensemble():
    """Create a Voting Ensemble with SVM, Random Forest, and XGBoost"""
    # Define the base models
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=2, random_state=42)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

    # Create the Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('rf', rf),
            ('xgb', xgb)
        ],
        voting='soft'
    )
    
    # Pipeline to include preprocessing
    pipeline = Pipeline([('voting', voting_clf)])
    
    return pipeline

def train_and_evaluate_voting_ensemble(X_train, X_test, y_train, y_test):
    """Train and evaluate the Voting Ensemble"""
    model = create_voting_ensemble()
    
    print("\nTraining Voting Ensemble...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nVoting Ensemble Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

if __name__ == "__main__":
    # Load data
    print("Loading training data...")
    train_data, train_labels = load_data(train_dir)

    print("Loading test data...")
    test_data, test_labels = load_data(test_dir)

    if len(train_data) == 0 or len(test_data) == 0:
        print("Error: No images could be loaded. Check your dataset path and image files.")
        exit(1)

    # Encode Labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    # Step 1: Fit the scaler on training data
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Scale training data
    X_val = scaler.transform(X_val)         # Scale validation data
    test_data = scaler.transform(test_data)  # Scale test data

    # Save the fitted scaler
    dump(scaler, 'scaler.joblib')
    print("Scaler saved as 'scaler.joblib'")

    # Step 2: Train and Evaluate Voting Ensemble
    best_model, accuracy = train_and_evaluate_voting_ensemble(X_train, test_data, y_train, test_labels)

    # Save the best model
    dump(best_model, 'voting_ensemble_skin_disease_model.joblib')
    print("Voting Ensemble Model saved as 'voting_ensemble_skin_disease_model.joblib'")