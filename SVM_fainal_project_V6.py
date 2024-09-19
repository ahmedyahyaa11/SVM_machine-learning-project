# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv('Bank_dataset.csv')  

# Display the column names and basic statistics of the dataset
print("Column names in the dataset:")
print(df.columns)

# Display first few rows and check for missing values
print(df.head())
print(df.isnull().sum())  # Check for missing values
print(df.describe())  # Summary statistics

# Select features and target for the model
features = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
target = df['accept']  # Target variable indicating acceptance

# Encode categorical features using LabelEncoder
label_encoders = {}  # To store the encoders for each categorical column
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])  # Encode categorical features
    label_encoders[column] = le  # Save encoder for potential future use

# Encode the target variable
target_encoder = LabelEncoder()
target = target_encoder.fit_transform(target)  # Encode target (binary classification)

# Standardize the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Scale features to have mean 0 and variance 1

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Initialize and train the SVM classifier
svm_clf = SVC(kernel='linear', probability=True, random_state=42)  # Using linear kernel with probability estimates
svm_clf.fit(X_train, y_train) 

# Make predictions on the test set
y_pred = svm_clf.predict(X_test)

# Evaluate the model using accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)  # Precision for positive class (default: 1)
recall = recall_score(y_test, y_pred, pos_label=1)  # Recall for positive class (default: 1)

# Print the evaluation metrics 
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate the ROC curve and AUC score
y_prob = svm_clf.predict_proba(X_test)[:, 1]  # Get the probability estimates for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Compute ROC curve
roc_auc = roc_auc_score(y_test, y_prob)  # Compute AUC score

# Plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# GUI Functionality
def predict_loan():
    try:
        # Collect user inputs
        age = int(entry_age.get())
        job = entry_job.get()
        marital = entry_marital.get()
        education = entry_education.get()
        default = entry_default.get()
        housing = entry_housing.get()
        loan = entry_loan.get()
        contact = entry_contact.get()
        month = entry_month.get()
        day_of_week = entry_day_of_week.get()
        duration = int(entry_duration.get())
        campaign = int(entry_campaign.get())
        pdays = int(entry_pdays.get())
        previous = int(entry_previous.get())
        poutcome = entry_poutcome.get()

        # Convert the input data into a DataFrame
        input_data = pd.DataFrame([[age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome]],
                                  columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])

        # Apply the same encoding and scaling as the training data
        for column in input_data.select_dtypes(include=['object']).columns:
            input_data[column] = label_encoders[column].transform(input_data[column])
        input_data_scaled = scaler.transform(input_data)
 
        # Make the prediction
        prediction = svm_clf.predict(input_data_scaled)

        # Decode the prediction (0: Reject, 1: Accept)
        result = target_encoder.inverse_transform(prediction)[0]

        # Display the result
        messagebox.showinfo("Prediction Result", f"The bank will {result} the loan.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the GUI using Tkinter
root = tk.Tk()
root.title("Bank Loan Approval Prediction")

# Adjust window size for better display
root.geometry("400x600")

# Create input fields with larger labels and entry boxes
tk.Label(root, text="Age:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5, sticky='w')
entry_age = tk.Entry(root, font=("Arial", 12), width=20)
entry_age.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Job:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=5, sticky='w')
entry_job = tk.Entry(root, font=("Arial", 12), width=20)
entry_job.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Marital Status:", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=5, sticky='w')
entry_marital = tk.Entry(root, font=("Arial", 12), width=20)
entry_marital.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Education:", font=("Arial", 12)).grid(row=3, column=0, padx=10, pady=5, sticky='w')
entry_education = tk.Entry(root, font=("Arial", 12), width=20)
entry_education.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Default:", font=("Arial", 12)).grid(row=4, column=0, padx=10, pady=5, sticky='w')
entry_default = tk.Entry(root, font=("Arial", 12), width=20)
entry_default.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Housing:", font=("Arial", 12)).grid(row=5, column=0, padx=10, pady=5, sticky='w')
entry_housing = tk.Entry(root, font=("Arial", 12), width=20)
entry_housing.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Loan:", font=("Arial", 12)).grid(row=6, column=0, padx=10, pady=5, sticky='w')
entry_loan = tk.Entry(root, font=("Arial", 12), width=20)
entry_loan.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Contact:", font=("Arial", 12)).grid(row=7, column=0, padx=10, pady=5, sticky='w')
entry_contact = tk.Entry(root, font=("Arial", 12), width=20)
entry_contact.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Month:", font=("Arial", 12)).grid(row=8, column=0, padx=10, pady=5, sticky='w')
entry_month = tk.Entry(root, font=("Arial", 12), width=20)
entry_month.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Day of Week:", font=("Arial", 12)).grid(row=9, column=0, padx=10, pady=5, sticky='w')
entry_day_of_week = tk.Entry(root, font=("Arial", 12), width=20)
entry_day_of_week.grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="Duration:", font=("Arial", 12)).grid(row=10, column=0, padx=10, pady=5, sticky='w')
entry_duration = tk.Entry(root, font=("Arial", 12), width=20)
entry_duration.grid(row=10, column=1, padx=10, pady=5)

tk.Label(root, text="Campaign:", font=("Arial", 12)).grid(row=11, column=0, padx=10, pady=5, sticky='w')
entry_campaign = tk.Entry(root, font=("Arial", 12), width=20)
entry_campaign.grid(row=11, column=1, padx=10, pady=5)

tk.Label(root, text="Pdays:", font=("Arial", 12)).grid(row=12, column=0, padx=10, pady=5, sticky='w')
entry_pdays = tk.Entry(root, font=("Arial", 12), width=20)
entry_pdays.grid(row=12, column=1, padx=10, pady=5)

tk.Label(root, text="Previous:", font=("Arial", 12)).grid(row=13, column=0, padx=10, pady=5, sticky='w')
entry_previous = tk.Entry(root, font=("Arial", 12), width=20)
entry_previous.grid(row=13, column=1, padx=10, pady=5)

tk.Label(root, text="Poutcome:", font=("Arial", 12)).grid(row=14, column=0, padx=10, pady=5, sticky='w')
entry_poutcome = tk.Entry(root, font=("Arial", 12), width=20)
entry_poutcome.grid(row=14, column=1, padx=10, pady=5)

# Create a larger predict button
predict_button = tk.Button(root, text="Predict Loan Approval", font=("Arial", 14), bg="green", fg="white", width=20, height=2, command=predict_loan)
predict_button.grid(row=15, columnspan=2, pady=20)

# Start the GUI loop
root.mainloop()
