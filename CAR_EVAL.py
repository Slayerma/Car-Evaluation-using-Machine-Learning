import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
class CarEvaluationDataset:
    def __init__(self, file_path):
        """
        Initialize the CarEvaluationDataset object.

        Args:
            file_path (str): The path to the file containing the dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the file is empty.
            pd.errors.ParserError: If the file contains invalid data.
        """
        try:
            self.data = pd.read_csv(file_path, header=None)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            print(f"File is empty: {file_path}")
            raise
        except pd.errors.ParserError:
            print(f"Invalid data in file: {file_path}")
            raise

        if self.data is None:
            raise ValueError("Data is None")

        if len(self.data.columns) != 7:
            raise ValueError("Data should have 7 columns")

        self.data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']

    def get_data(self):
        """Return the dataset.

        Returns:
            pd.DataFrame: The dataset.

        Raises:
            AttributeError: If the dataset is None.
        """
        if self.data is None:
            raise AttributeError("Data is None")
        return self.data

    def get_missing_values(self):
        """Check for missing values in the dataset.

        Returns:
            pd.Series: A series with the number of missing values in each column.
        """
        if self.data is None:
            raise AttributeError("Data is None")

        return self.data.isnull().sum()

    def get_value_counts(self):
        """Print the value counts for each column.

        Raises:
            AttributeError: If the dataset is None.
        """
        if self.data is None:
            raise AttributeError("Data is None")

        for column in self.data.columns:
            print(self.data[column].value_counts())


class DataVisualizer:
    def __init__(self, data):
        if data is None:
            raise AttributeError("Data is None")
        self.data = data

    def plot_bar_chart(self, column):
        """Plot a bar chart of the distribution of values in a column.

        Args:
            column (str): The name of the column to plot.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        plt.figure(figsize=(6,4))
        sns.countplot(self.data[column], palette="viridis")
        plt.title(f'Distribution of {column}')
        plt.show()


class CustomLabelEncoder:
    def __init__(self) -> None:
        """
        Initialize the CustomLabelEncoder object.
        """
        self.le = LabelEncoder()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the label encoder to the given data and transform it.

        Args:
            data (pandas.DataFrame): The data to fit and transform.

        Returns:
            pandas.DataFrame: The transformed data.

        Raises:
            AttributeError: If the data is None.
            ValueError: If a column in the data is null.
        """
        if data is None:
            raise AttributeError("Data is None")

        for column in data.columns:
            if data[column].isnull().any():
                raise ValueError(f"Column '{column}' contains null values")

            data[column] = self.le.fit_transform(data[column])

        return data
    def plot_correlation_heatmap(data):
        """Plot a correlation heatmap of the dataset features."""
        plt.figure(figsize=(10, 8))
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()



 

from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

class CustomDecisionTreeClassifier:
    def __init__(self, random_state=42):
        if random_state is None:
            raise AttributeError("Random state is None")
        self.clf = SklearnDecisionTreeClassifier(random_state=random_state)

    def fit(self, X, y):
        """
        Fit the CustomDecisionTreeClassifier to the given data.

        Args:
            X (pandas.DataFrame): The features of the data.
            y (pandas.Series): The target of the data.

        Raises:
            AttributeError: If the data or target is None.
            ValueError: If a column in the data or target is null.
        """
        if X is None or y is None:
            raise AttributeError("Data or target is None")

        for column in X.columns:
            if X[column].isnull().any():
                raise ValueError(f"Column '{column}' in data contains null values")

        if y.isnull().any():
            raise ValueError("Target contains null values")

        self.clf.fit(X, y)

    def predict(self, X):
        """
        Predict the target of the given data.

        Args:
            X (pandas.DataFrame): The features of the data.

        Raises:
            AttributeError: If the data is None.
            ValueError: If a column in the data is null.

        Returns:
            numpy.ndarray: The predicted target.
        """
        if X is None:
            raise AttributeError("Data is None")

        for column in X.columns:
            if X[column].isnull().any():
                raise ValueError(f"Column '{column}' in data contains null values")

        return self.clf.predict(X)

from sklearn.tree import plot_tree

def plot_decision_tree(clf, feature_names):
    """Plot the decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=['unacc', 'acc', 'good', 'vgood'])
    plt.title('Decision Tree Visualization')
    plt.show()





from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

class CustomRandomForestClassifier:
    def __init__(self, random_state=42):
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        self.clf = SklearnRandomForestClassifier(random_state=random_state)

    def fit(self, X, y):
        """
        Fit the CustomRandomForestClassifier to the training data.

        Args:
            X (pandas.DataFrame): The features of the training data.
            y (pandas.DataFrame or numpy.ndarray): The target of the training data.

        Raises:
            AttributeError: If X or y is None.
            ValueError: If a column in X contains null values.
        """
        if X is None or y is None:
            raise AttributeError("X and y must not be None")

        for column in X.columns:
            if X[column].isnull().any():
                raise ValueError(f"Column '{column}' in X contains null values")

        self.clf.fit(X, y)

    def predict(self, X):
        """
        Predict the target of the given features.

        Args:
            X (pandas.DataFrame): The features of the data.

        Raises:
            AttributeError: If X is None.
            ValueError: If a column in X contains null values.

        Returns:
            numpy.ndarray: The predicted target.
        """
        if X is None:
            raise AttributeError("X must not be None")

        for column in X.columns:
            if X[column].isnull().any():
                raise ValueError(f"Column '{column}' in X contains null values")

        return self.clf.predict(X)


from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
class CustomXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        self.clf = XGBClassifier(*args, **kwargs)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def get_params(self, deep=True):
        return self.clf.get_params(deep)

    def set_params(self, **params):
        self.clf.set_params(**params)
        return self

class CustomGridSearchCV:
    def __init__(self, estimator, param_grid, cv=3, n_jobs=-1):
        self.grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=n_jobs)

    def fit(self, X, y):
        self.grid_search.fit(X, y)

    def get_best_params(self):
        return self.grid_search.best_params_

    def get_best_model(self):
        return self.grid_search.best_estimator_
    def display_grid_search_results(grid_search):
        print("Best Parameters:", grid_search.get_best_params())
        print("Best Score:", grid_search.grid_search.best_score_)

def plot_correlation_heatmap(data):
    """Plot a correlation heatmap of the dataset features."""
    # Encode categorical features to numeric
    data_encoded = data.copy()
    for column in data_encoded.columns:
        if data_encoded[column].dtype == 'object':
            data_encoded[column] = data_encoded[column].astype('category').cat.codes

    plt.figure(figsize=(10, 8))
    corr_matrix = data_encoded.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


def plot_class_distribution(y, title):
    """Plot the class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="viridis")
    plt.title(title)
    plt.show()



def display_grid_search_results(grid_search):
    """Display the results of the grid search."""
    print("Best Parameters:", grid_search.get_best_params())
    print("Best Score:", grid_search.grid_search.best_score_)



class CustomStackingClassifier:
    def __init__(self, estimators, final_estimator):
        self.clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
   
    def plot_class_distribution(y, title):
        """Plot the class distribution."""
        plt.figure(figsize=(6, 4))
        sns.countplot(y, palette="viridis")
        plt.title(title)
        plt.show()





def main():
    # Load the dataset
    dataset = CarEvaluationDataset('/Users/syedmohathashimali/Downloads/car+evaluation/car.data')

    # Get the data
    data = dataset.get_data()

    # Get the missing values
    missing_values = dataset.get_missing_values()
    print(missing_values)

    # Get the value counts
    value_counts = dataset.get_value_counts()

    # Plot bar charts
    visualizer = DataVisualizer(data)
    for column in data.columns[:-1]:
        visualizer.plot_bar_chart(column)

    # Plot correlation heatmap
    plot_correlation_heatmap(data)

    # Encode labels
    label_encoder = CustomLabelEncoder()
    data = label_encoder.fit_transform(data)

    # Split the data
    X = data.drop('acceptability', axis=1)
    y = data['acceptability']

    # Plot class distribution before SMOTE
    plot_class_distribution(y, 'Class Distribution Before SMOTE')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the CustomDecisionTreeClassifier
    dt_clf = CustomDecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred_dt = dt_clf.predict(X_test)
    print("CustomDecisionTreeClassifier Accuracy:", accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt))

    # Plot Decision Tree
    plot_decision_tree(dt_clf.clf, X.columns)

    # Train the RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred_rf = rf_clf.predict(X_test)
    print("RandomForestClassifier Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # Define parameter grid for GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Create GridSearchCV instance
    grid_search = CustomGridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=3, n_jobs=-1)

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Print best parameters
    display_grid_search_results(grid_search)

    # Use the best model for predictions
    y_pred_tuned = grid_search.get_best_model().predict(X_test)
    print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
    print(classification_report(y_test, y_pred_tuned))
    
    # Train the XGBClassifier
    xgb_clf = CustomXGBClassifier()
    xgb_clf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred_xgb = xgb_clf.predict(X_test)
    print("XGBClassifier Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))

    # Feature engineering: interaction terms
    X_interaction = pd.DataFrame({
        'safety_high_buying_vhigh': (X['safety'] == 'high') & (X['buying'] == 'vhigh'),
        'safety_med_buying_high': (X['safety'] == 'med') & (X['buying'] == 'high'),
        'safety_low_buying_low': (X['safety'] == 'low') & (X['buying'] == 'low')
    })
    X = pd.concat([X, X_interaction], axis=1)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Plot class distribution after SMOTE
    plot_class_distribution(y_train_smote, 'Class Distribution After SMOTE')

    # Train XGBoost with SMOTE
    xgb_clf_smote = CustomXGBClassifier()
    xgb_clf_smote.fit(X_train_smote, y_train_smote)

    # Evaluate the model
    y_pred_smote = xgb_clf_smote.predict(X_test)
    print("XGBClassifier with SMOTE Accuracy:", accuracy_score(y_test, y_pred_smote))
    print(classification_report(y_test, y_pred_smote))

    # Stacking Ensemble
    estimators = [('xgb', xgb_clf_smote), ('rf', rf_clf)]
    stacking_clf = CustomStackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking_clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred_stack = stacking_clf.predict(X_test)
    print("Stacking Ensemble Accuracy:", accuracy_score(y_test, y_pred_stack))
    print(classification_report(y_test, y_pred_stack))

   

if __name__ == '__main__':
    main()
