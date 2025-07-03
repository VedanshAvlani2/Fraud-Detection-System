import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("creditcard.csv")
print("\U0001F50D Dataset Shape:", df.shape)
print("\U0001F9FE Class Distribution:\n", df['Class'].value_counts())

# ----------------------------
# Preprocessing
# ----------------------------
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df = df.dropna()

features = [col for col in df.columns if col != 'Class']
X_all = df[features]
y_all = df['Class']

# ----------------------------
# Scheduled Retraining Simulation (Optimized)
# ----------------------------
print("\n\U0001F4C6 Optimized Retraining Simulation Begins...\n")

results = []
step_values = np.linspace(df['Time'].min(), df['Time'].max(), 30, dtype=int)
window_size = 15000

for step in step_values:
    temp_df = df[(df['Time'] <= step) & (df['Time'] > (step - window_size))]

    if temp_df['Class'].nunique() < 2 or len(temp_df) < 300:
        continue

    X_temp = temp_df[features]
    y_temp = temp_df['Class']

    scaler = StandardScaler()
    X_scaled_temp = scaler.fit_transform(X_temp)

    if y_temp.value_counts().min() < 2:
        continue

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
        )
    except:
        continue

    smote = SMOTE(random_state=42, k_neighbors=1)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    except ValueError:
        continue

    model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\U0001F501 Step {step}: Fraud Cases: {y_temp.sum():,}, AUC: {auc:.3f}")

    results.append({
        'step': step,
        'auc': auc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score']
    })

# ----------------------------
# Future-Time Fraud Detection Simulation
# ----------------------------
print("\n⏳ Simulating Future-Time Fraud Detection...\n")

cutoff_time = df['Time'].quantile(0.80)
train_df = df[df['Time'] <= cutoff_time]
test_df = df[df['Time'] > cutoff_time]

print(f"📌 Training on {len(train_df):,} rows (Time <= {cutoff_time:.0f})")
print(f"📌 Testing  on {len(test_df):,} rows (Time >  {cutoff_time:.0f})")

X_train = train_df[features]
y_train = train_df['Class']
X_test = test_df[features]
y_test = test_df['Class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if y_train.value_counts().min() >= 2:
    smote = SMOTE(random_state=42, k_neighbors=1)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    except ValueError:
        print("⚠️ SMOTE failed due to class imbalance.")
        X_resampled, y_resampled = X_train_scaled, y_train
else:
    print("⚠️ Skipping SMOTE due to low fraud samples.")
    X_resampled, y_resampled = X_train_scaled, y_train

model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print("\n📊 Future-Time Test Evaluation:")
print(f"  - Fraud Cases: {y_test.sum():,}")
print(f"  - AUC Score  : {auc:.3f}")
print(f"  - Precision  : {report['1']['precision']:.2f}")
print(f"  - Recall     : {report['1']['recall']:.2f}")
print(f"  - F1-Score   : {report['1']['f1-score']:.2f}")

# ----------------------------
# Threshold Tuning
# ----------------------------
print("\n🛠️ Threshold Tuning:")
thresholds = [0.5, 0.4, 0.3, 0.2]
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)

    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Threshold = {thresh:>4} | AUC = {auc:.3f} | Precision = {precision:.2f} | Recall = {recall:.2f} | F1 = {f1:.2f}")

# ----------------------------
# Plot AUC over time
# ----------------------------
if results:
    results_df = pd.DataFrame(results)
    print("\n📊 Results preview:\n", results_df.head())

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results_df, x='step', y='auc', marker='o')
    plt.title("AUC Score Over Retraining Steps")
    plt.xlabel("Step")
    plt.ylabel("AUC Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No retraining results found. Please check data or loop logic.")
