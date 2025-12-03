import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import urllib.parse 

# --- Data Loading, Preprocessing, and Feature Engineering ---

# Load Data (Ensure these CSV files are in the same directory as your script)
data1 = pd.read_csv('Project/urldata.csv')
data2 = pd.read_csv('Project/malicious_phish.csv')

# Preprocessing: Map 'type' to binary 'result'
mapping = {'phishing' : 'malicious', 'defacement' : 'malicious','malware' : 'malicious',}
data2['type'] = data2['type'].replace(mapping)
mapping2 = {'malicious': 1, 'benign': 0}
data2['result'] = data2['type'].replace(mapping2).astype(int)

# Drop unused columns and combine
data1.drop(['Unnamed: 0', 'label'],axis=1, inplace=True)
data2.drop(['type'],axis=1, inplace=True)
data = pd.concat([data1, data2],ignore_index=True)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True,drop=True)

# Under-sampling to balance classes (Original Code)
rus = RandomUnderSampler(random_state=42)
x_url = data[['url']]
y = data['result']
x_resampled, y_resampled = rus.fit_resample(x_url,y)
data = pd.concat([x_resampled,y_resampled],axis=1)
data.reset_index(inplace=True,drop=True)

# Original Feature: URL Length
data['char'] = data['url'].str.len()

# Original Feature: Number of Queries
def count_query_params(url):
    try:
        query_string = urllib.parse.urlparse(url).query
        if not query_string:
            return 0
        return len(query_string.split('&'))
    except:
        return 0
data['queries'] = data['url'].apply(count_query_params).astype(int)

# NEW FEATURE: Number of Subdomains
def count_subdomains(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        netloc = urllib.parse.urlparse(url).netloc
        hostname = netloc.split(':')[0]
        return hostname.count('.')
    except:
        return 0
data['num_subdomains'] = data['url'].apply(count_subdomains)


# --- Train/Val/Test Split (UPDATED FEATURE SET) ---

# Define feature set 'x' now including the new feature
x = data[['char', 'queries', 'num_subdomains']]
y = data['result']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}, Test samples: {len(x_test)}")

# --- StandardScaler (CORRECTED for no data leakage) ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val) # Corrected: only transform
x_test_scaled = scaler.transform(x_test) # Corrected: only transform


# --- Model Training (Faster LinearSVC) ---
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X=x_train_scaled, y=y_train)


# --- Evaluation ---
print("\n--- Model Evaluation (LinearSVC with 'num_subdomains') ---")

# Validation Set Evaluation
val_pred = svm.predict(x_val_scaled)
print("\nValidation Results:")
print("Accuracy:", accuracy_score(y_val, val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))
print("Classification Report:\n", classification_report(y_val, val_pred))

# Test Set Evaluation
test_pred = svm.predict(x_test_scaled)
print("\nTest Results:")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("Classification Report:\n", classification_report(y_test, test_pred))
