import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class TemporalSITS:
    """
    A simplified Python implementation of Satellite Image Time Series (SITS) analysis
    that focuses only on the temporal dimension, ignoring location data.
    """
    
    def __init__(self):
        """
        Initialize the TemporalSITS analysis object.
        """
        self.model = None
        self.training_data = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_from_dataframe(self, df, time_col='time', band_cols=None, label_col=None):
        """
        Load time series data from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing time series
            time_col (str): Name of the time column
            band_cols (list): List of band/feature column names
            label_col (str): Name of the label column (if available)
        """
        if band_cols is None:
            # Assume all columns except time and label are bands
            band_cols = [col for col in df.columns if col not in [time_col, label_col]]
            
        self.feature_names = band_cols
        
        if label_col and label_col in df.columns:
            # Prepare training data
            self.training_data = {
                'times': df[time_col].values,
                'features': df[band_cols].values,
                'labels': self.label_encoder.fit_transform(df[label_col].values)
            }
        else:
            # Just store features for prediction
            self.training_data = {
                'times': df[time_col].values,
                'features': df[band_cols].values
            }
    
    def load_from_array(self, X, y=None, timestamps=None, feature_names=None):
        """
        Load time series data from numpy arrays.
        
        Args:
            X (np.ndarray): 2D array of shape (n_samples, n_features)
            y (np.ndarray): 1D array of labels (optional)
            timestamps (np.ndarray): 1D array of timestamps (optional)
            feature_names (list): List of feature/band names (optional)
        """
        self.feature_names = feature_names or [f'band_{i}' for i in range(X.shape[1])]
        
        if y is not None:
            self.training_data = {
                'times': timestamps,
                'features': X,
                'labels': self.label_encoder.fit_transform(y)
            }
        else:
            self.training_data = {
                'times': timestamps,
                'features': X
            }
    
    def plot_temporal_profile(self, sample_idx=0):
        """
        Plot the temporal profile for a given sample.
        
        Args:
            sample_idx (int): Index of the sample to plot
        """
        if self.training_data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if 'times' not in self.training_data or self.training_data['times'] is None:
            # Create dummy time axis if not provided
            times = np.arange(self.training_data['features'].shape[1])
        else:
            times = self.training_data['times']
            
        features = self.training_data['features'][sample_idx]
        
        plt.figure(figsize=(10, 5))
        
        if len(features.shape) == 1:
            # Single time series
            plt.plot(times, features, marker='o', linestyle='-')
            plt.title(f'Temporal profile for sample {sample_idx}')
        else:
            # Multiple bands
            for i, band in enumerate(self.feature_names):
                plt.plot(times, features[:, i], marker='o', linestyle='-', label=band)
            plt.title(f'Temporal profile for sample {sample_idx}')
            plt.legend()
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train a Random Forest classifier on the time series data.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if self.training_data is None or 'labels' not in self.training_data:
            raise ValueError("No labeled training data available.")
            
        X = self.training_data['features']
        y = self.training_data['labels']
        
        # If we have 3D data (n_samples, n_timesteps, n_features), flatten time dimension
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten to (n_samples, n_timesteps*n_features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        print("Model evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
    
    def predict(self, X):
        """
        Predict class labels for new time series data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("No model trained. Please train a model first.")
            
        # Flatten if 3D input
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
            
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def get_feature_importances(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            pd.DataFrame: DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("No model trained. Please train a model first.")
            
        if len(self.model.feature_importances_) == len(self.feature_names):
            # Simple case (no time dimension)
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # Flattened time series features
            n_timesteps = self.model.feature_importances_.shape[0] // len(self.feature_names)
            importance_df = pd.DataFrame({
                'timestep': np.repeat(np.arange(n_timesteps), len(self.feature_names)),
                'band': np.tile(self.feature_names, n_timesteps),
                'importance': self.model.feature_importances_
            })
            return importance_df.pivot_table(index='timestep', columns='band', values='importance')
        

class SITSAnalysis:
    """
    A Python implementation of Satellite Image Time Series (SITS) analysis
    for Earth Observation data cubes.
    """
    
    def __init__(self, data_cube=None, random_state=42):
        """
        Initialize the SITS analysis object.
        
        Args:
            data_cube (xarray.Dataset): Input data cube with dimensions (time, y, x, bands)
        """
        self.data_cube = data_cube
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.training_data = None
        
    def load_data_cube(self, file_path, engine='netcdf4'):
        """
        Load a data cube from a file (NetCDF, Zarr, etc.)
        
        Args:
            file_path (str): Path to the data cube file
            engine (str): Engine to use for reading the file
        """
        self.data_cube = xr.open_dataset(file_path, engine=engine)
        
    def get_temporal_profile(self, longitude, latitude, band=None):
        """
        Extract the temporal profile for a given location.
        
        Args:
            longitude (float): Longitude coordinate
            latitude (float): Latitude coordinate
            band (str, optional): Specific band to extract. If None, all bands are returned.
            
        Returns:
            xarray.DataArray: Temporal profile for the location
        """
        if self.data_cube is None:
            raise ValueError("No data cube loaded. Please load a data cube first.")
            
        # Find nearest pixel
        lon_idx = np.abs(self.data_cube.lon - longitude).argmin()
        lat_idx = np.abs(self.data_cube.lat - latitude).argmin()
        
        if band:
            return self.data_cube[band].isel(lon=lon_idx, lat=lat_idx)
        else:
            return self.data_cube.isel(lon=lon_idx, lat=lat_idx)
        
    def plot_temporal_profile(self, longitude, latitude, band=None):
        """
        Plot the temporal profile for a given location.
        
        Args:
            longitude (float): Longitude coordinate
            latitude (float): Latitude coordinate
            band (str, optional): Specific band to plot. If None, all bands are plotted.
        """
        profile = self.get_temporal_profile(longitude, latitude, band)
        
        plt.figure(figsize=(10, 5))
        if band:
            profile.plot(marker='o', linestyle='-')
            plt.title(f'Temporal profile at ({longitude}, {latitude}) - Band: {band}')
        else:
            for b in profile.data_vars:
                profile[b].plot(marker='o', linestyle='-', label=b)
            plt.title(f'Temporal profile at ({longitude}, {latitude})')
            plt.legend()
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
        
    def add_training_data(self, samples):
        """
        Add training data to the SITS object.
        
        Args:
            samples (list of dicts): Each dict should contain:
                - 'longitude': longitude coordinate
                - 'latitude': latitude coordinate
                - 'label': class label
                - 'start_date': start date of sample (optional)
                - 'end_date': end date of sample (optional)
        """
        features = []
        labels = []
        
        for sample in samples:
            profile = self.get_temporal_profile(sample['longitude'], sample['latitude'])
            
            # Convert to pandas DataFrame and add label
            df = profile.to_dataframe()
            df['label'] = sample['label']
            
            features.append(df.drop(columns=['label']).values.flatten())
            labels.append(sample['label'])
            
        self.training_data = {
            'features': np.array(features),
            'labels': np.array(labels)
        }
        
    def train_model(self,test_size=0.2, random_state=42):
        """
        Train a Random Forest classifier on the training data.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if self.training_data is None:
            raise ValueError("No training data available. Please add training data first.")
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_data['features'],
            self.training_data['labels'],
            test_size=test_size,
            random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        print("Model evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
    def classify_image(self, output_band='classification'):
        """
        Classify the entire data cube using the trained model.
        
        Args:
            output_band (str): Name of the output band to store classification results
            
        Returns:
            xarray.Dataset: Data cube with classification results added
        """
        if self.model is None:
            raise ValueError("No model trained. Please train a model first.")
            
        # Reshape the data cube for prediction
        original_shape = self.data_cube.to_array().shape
        n_bands = original_shape[0]
        n_times = original_shape[1]
        
        # Stack all bands and times for each pixel
        stacked = self.data_cube.to_array().values
        stacked = stacked.transpose(1, 2, 3, 0)  # (time, y, x, bands)
        stacked = stacked.reshape(-1, n_times * n_bands)
        
        # Predict
        predictions = self.model.predict(stacked)
        predictions = predictions.reshape(self.data_cube.dims['lat'], self.data_cube.dims['lon'])
        
        # Add to data cube
        self.data_cube[output_band] = (('lat', 'lon'), predictions)
        
        return self.data_cube