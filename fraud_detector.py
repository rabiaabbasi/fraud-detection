import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

class FraudDetector:
    """
    A reusable class to detect fraud in financial transaction data.
    Covers loading, preprocessing, training, and evaluation.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def load_data(self):
        """Load the CSV dataset and show basic info."""
        self.df = pd.read_csv(self.filepath)
        print("Data loaded successfully!")
        print("Shape: ", self.df.shape)
        print("\nFirst 5 rows:\n", self.df.head())
        print("\nColumn names: ", list(self.df.columns))
        return self.df

    def explore_data(self):
        """Show basic statistics and missing value info."""
        print("\nMissing Values: ")
        print(self.df.isnull().sum())

        print("\nData Types: ")
        print(self.df.dtypes)

        print("\nBasic Statistics: ")
        print(self.df.describe())

    def preprocess(self, target_column):
        """
        Clean data, encode categories, handle imbalance with SMOTE,
        and split into train/test sets.
        """
        df = self.df.copy()

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Encode any text/categorical columns
        le = LabelEncoder()
        for col in df.select_dtypes(include= 'object' ).columns:
            if col != target_column:
                df[col] = le.fit_transform(df[col])

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print("\nClass distribution before SMOTE:\n", y.value_counts())

        # Apply SMOTE to fix imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print("\nClass distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        print("\nTraining samples: ", self.X_train.shape[0])
        print("Testing samples:  ", self.X_test.shape[0])

    def train(self):
        """Train Logistic Regression and Random Forest models."""

        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        print("Done!")

        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("Done!")

    def evaluate(self):
        """Print classification report and plot confusion matrix for each model."""

        for name, model in self.models.items():
            print(f"\n{'='*45}")
            print(f"  Results for: {name}")
            print(f"{'='*45}")

            y_pred = model.predict(self.X_test)

            print(classification_report(self.y_test, y_pred))

            # Confusion matrix plot
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix — " , name)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f'plots/confusion_matrix_{name.replace(" ", "_")}.png')
            plt.show()
            print("Confusion matrix saved to plots/")

    def plot_feature_importance(self):
        """Plot feature importance from the Random Forest model."""

        rf_model = self.models.get('Random Forest')
        if rf_model is None:
            print("Train the model first using .train()")
            return

        importances = rf_model.feature_importances_
        feature_names = self.X_train.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues_r')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.show()
        print("Feature importance chart saved to plots/")