import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import torchio as tio
import shap
from sklearn.ensemble import RandomForestClassifier

class CTAugmenter:
    def __init__(self):
        self.transform = tio.Compose([
            tio.RandomFlip(axes=(0, 1), flip_probability=0.5),  # Random horizontal/vertical flip
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=(5, 5, 3)),  # Rotate ±10°, randomly shift (x, y, z)
            tio.RandomGamma(log_gamma=(0.8, 1.2))  # Randomly adjust brightness (gamma value)
        ])

    def __call__(self, image):

        # Convert numpy.ndarray to torch.Tensor
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
            # Perform data augmentation
            image = self.transform(image)  # Apply predefined transformation operations
        return image


class MultimodalDataset(Dataset):
    def __init__(self,
                 data_files,  # List of CSV file paths for clinical data
                 image_dirs,  # List of directories that may contain CT images
                 mask_dir,  # Directory containing mask images
                 target_image_count,  # Target number of slices
                 img_dimensions,  # Target image dimensions (H, W)
                 selected_feature_count,  # Number of selected clinical features
                 image_feature_list,  # List of image feature CSVs
                 selected_image_feature_count,  # Number of selected image features
                 transform=None,
                 augment=None,
                 scaler=None,
                 selector=None,
                 image_scaler=None,
                 image_selector=None):
        """
        :param data_files: CSV file path list
        :param image_dirs: Multiple directories (list) where CT images may be located
        :param mask_dir: Mask Image Directory
        :param target_image_count: Target slice number
        :param img_dimensions: Target image size (H, W)
        """
        if not isinstance(data_files, list):
            raise ValueError("`data_files` Must be a list of CSV file paths")

        # Read all clinical characteristics CSV files and merge
        data_list = []
        for file in data_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
            df = pd.read_csv(file)
            data_list.append(df)

        # Read the imaging feature CSV file and merge
        image_data_list = []
        for image_file in image_feature_list:
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"File not found: {image_file}")
            image_df = pd.read_csv(image_file)
            image_data_list.append(image_df)


        self.data = pd.concat(data_list, ignore_index=True).drop_duplicates()
        self.image_feature_list = image_feature_list
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.mask_dir = mask_dir
        self.image_dimensions = img_dimensions
        self.target_image_count = target_image_count
        self.transform = transform
        self.augment = augment
        self.image_feature_data = pd.concat(image_data_list, ignore_index=True).drop_duplicates()
        self.selected_feature_count = selected_feature_count
        self.selected_image_feature_count = selected_image_feature_count

        # Clinical feature data cleaning(if you have)
        # if 'Volume of hematoma at primary site on admission*' in self.data.columns:
        #     self.data['Volume of hematoma at primary site on admission*'] = self.data['Volume of hematoma at primary site on admission*'].astype(str).str.replace('ml', '').astype(float)

        self.data_clean = self.data.copy()
        self.data_clean.fillna(self.data_clean.median(numeric_only=True), inplace=True)


        X = self.data_clean.iloc[:, 2:]
        y = self.data_clean.iloc[:, 1]

        # Clinical feature standardization and feature selection
        if scaler is None and selector is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.selector = SelectKBest(f_classif, k=selected_feature_count)
            self.X_selected = self.selector.fit_transform(X_scaled, y)
        else:
            self.scaler = scaler
            self.selector = selector
            X_scaled = self.scaler.transform(X)
            self.X_selected = self.selector.transform(X_scaled)
            feature_names = X.columns

            mask = self.selector.get_support()

            selected_feature_names = feature_names[mask]
        self.y = y

        # Imaging feature data cleaning
        self.image_feature_data_clean = self.image_feature_data.copy()
        self.image_feature_data_clean.fillna(self.image_feature_data_clean.mean(numeric_only=True), inplace=True)
        image_feature_X = self.image_feature_data_clean.iloc[:, 2:]
        image_feature_y = self.image_feature_data_clean.iloc[:, 1]
        # Imaging feature standardization and feature selection(if you have)
        if image_scaler is None and image_selector is None:
            self.image_scaler = StandardScaler()
            image_feature_X_scaled = self.image_scaler.fit_transform(image_feature_X)
            self.image_selector = SelectKBest(f_classif, k=selected_image_feature_count)
            self.X_selected_image = self.image_selector.fit_transform(image_feature_X_scaled, image_feature_y)
        else:
            self.image_scaler = image_scaler
            self.image_selector = image_selector
            image_feature_X_scaled = self.image_scaler.transform(image_feature_X)
            self.X_selected_image = self.image_selector.transform(image_feature_X_scaled)
            feature_names = image_feature_X.columns
            mask = self.image_selector.get_support()


    def __len__(self):
        return len(self.data_clean)

    def find_patient_folder(self, patient_name):
        """ Find patient folders in multiple `image_dirs` directories """
        for image_dir in self.image_dirs:
            potential_path = os.path.join(image_dir, patient_name)
            if os.path.exists(potential_path):
                return potential_path
        return None

    def load_dicom_series(self, folder_path, target_image_count, image_dimensions):
        # Collect all DICOM file paths
        dicom_paths = []
        for root, _, files in os.walk(folder_path):
            dicom_paths.extend([os.path.join(root, f) for f in files if f.lower().endswith('.dcm')])

        if not dicom_paths:
            raise RuntimeError(f"{folder_path} no DICOM")

        slices = []
        for path in dicom_paths:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                slice_pos = float(getattr(ds, 'SliceLocation', 0))
                instance_num = int(getattr(ds, 'InstanceNumber', 0))
                image_pos = list(map(float, getattr(ds, 'ImagePositionPatient', [0, 0, 0])))

                slices.append({
                    'path': path,
                    'instance_number': instance_num,
                    'slice_location': slice_pos,
                    'image_position': image_pos
                })
            except Exception as e:
                raise RuntimeError(f"fail to read DICOM: {path} - {str(e)}")

        # Multi-level sorting strategy (first by spatial position, then by instance number)
        try:
            slices.sort(key=lambda s: (s['image_position'][2], s['instance_number']))
        except:
            slices.sort(key=lambda s: (s['slice_location'], s['instance_number']))

        # Load pixel data and process it
        all_images = []
        for slice_info in slices:
            ds = pydicom.dcmread(slice_info['path'])
            img = ds.pixel_array.astype(np.float32)

            rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
            hu_img = img * rescale_slope + rescale_intercept

            hu_img = np.clip(hu_img, -100, 300)

            img_resized = cv2.resize(hu_img, image_dimensions)
            all_images.append(img_resized)

        all_images = np.array(all_images, dtype=np.float32)

        # Interpolation Fill
        if len(all_images) < target_image_count:
            num_frames, height, width = all_images.shape
            x = np.linspace(0, num_frames - 1, num_frames)
            x_new = np.linspace(0, num_frames - 1, target_image_count)

            interpolated_images = np.zeros((target_image_count, height, width))

            for i in range(height):
                for j in range(width):
                    interpolator = interp1d(x, all_images[:, i, j], kind='linear', fill_value="extrapolate")
                    interpolated_images[:, i, j] = interpolator(x_new)

            images = interpolated_images

        # Resampling
        elif len(all_images) > target_image_count:
            indices = np.linspace(0, len(all_images) - 1, target_image_count).astype(int)
            images = all_images[indices]

        else:
            images = all_images

        return np.stack(images, axis=0)

    def load_mask_series(self, mask_folder):
        mask_slices = []

        for file_name in os.listdir(mask_folder):
            if not file_name.endswith('.dcm'):
                continue
            path = os.path.join(mask_folder, file_name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                slice_pos = float(getattr(ds, 'SliceLocation', 0))
                instance_num = int(getattr(ds, 'InstanceNumber', 0))
                image_pos = list(map(float, getattr(ds, 'ImagePositionPatient', [0, 0, 0])))

                mask_slices.append({
                    'path': path,
                    'instance_number': instance_num,
                    'slice_location': slice_pos,
                    'image_position': image_pos
                })
            except Exception as e:
                raise RuntimeError(f"Failed to read mask DICOM metadata: {path} - {str(e)}")

        try:
            mask_slices.sort(key=lambda s: (s['image_position'][2], s['instance_number']))
        except:
            mask_slices.sort(key=lambda s: (s['slice_location'], s['instance_number']))

        all_masks = []
        for slice_info in mask_slices:
            ds = pydicom.dcmread(slice_info['path'])
            mask = ds.pixel_array.astype(np.float32)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-6)
            mask_resized = cv2.resize(mask, self.image_dimensions)
            all_masks.append(mask_resized)

        return np.array(all_masks, dtype=np.float32)

    def resample_images(self, images):
        num_frames, height, width = images.shape
        if num_frames < self.target_image_count:
            x = np.linspace(0, num_frames - 1, num_frames)
            x_new = np.linspace(0, num_frames - 1, self.target_image_count)
            interpolated_images = np.zeros((self.target_image_count, height, width))
            for i in range(height):
                for j in range(width):
                    interpolator = interp1d(x, images[:, i, j], kind='linear', fill_value="extrapolate")
                    interpolated_images[:, i, j] = interpolator(x_new)
            return interpolated_images
        elif num_frames > self.target_image_count:
            indices = np.linspace(0, num_frames - 1, self.target_image_count).astype(int)
            return images[indices]
        else:
            return images

    def __getitem__(self, idx):
        patient_name = self.data_clean.iloc[idx, 0]
        label = int(self.y.iloc[idx])

        ct_folder_path = self.find_patient_folder(patient_name)
        if ct_folder_path is None:
            raise FileNotFoundError(f"CT image folder not found for patient {patient_name}")

        ct_images = self.load_dicom_series(folder_path=ct_folder_path,
                                           target_image_count=self.target_image_count,
                                           image_dimensions=self.image_dimensions)

        mask_folder_path = None
        if isinstance(self.mask_dir, list):  # Check if mask_dir is a list
            for dir_path in self.mask_dir:
                potential_path = os.path.join(dir_path, patient_name)
                if os.path.exists(potential_path):
                    mask_folder_path = potential_path
                    break
        else:  # If it's a single directory path
            mask_folder_path = os.path.join(self.mask_dir, patient_name)

        if mask_folder_path is None or not os.path.exists(mask_folder_path):
            raise FileNotFoundError(f"No mask folder found for patient {patient_name}")

        mask_images = self.load_mask_series(mask_folder_path)
        mask_images = self.resample_images(mask_images)

        ct_images = np.expand_dims(ct_images, axis=1)
        mask_images = np.expand_dims(mask_images, axis=1)

        if self.augment:
            augmenter = CTAugmenter()
            ct_images = augmenter(ct_images)
            mask_images = augmenter(mask_images)

        if self.transform:
            ct_images = self.transform(ct_images)
            mask_images = self.transform(mask_images)

        structured_data = torch.tensor(self.X_selected[idx], dtype=torch.float32)

        matched_row = self.image_feature_data_clean[self.image_feature_data_clean.iloc[:, 0] == patient_name]
        if matched_row.empty:
            raise ValueError(f"No imaging features found for patient {patient_name} ")


        patient_image_features = self.image_feature_data_clean
        matched_row = patient_image_features[patient_image_features.iloc[:, 0] == patient_name]
        if matched_row.empty:
            raise ValueError(f"No imaging features found for patient {patient_name}")
        matched_row = matched_row.iloc[0:1, 2:]  # Take only the features (remove the first two columns: name and label)
        matched_row.fillna(matched_row.mean(numeric_only=True), inplace=True)
        image_features_scaled = self.image_scaler.transform(matched_row)
        image_features_selected = self.image_selector.transform(image_features_scaled)
        image_features_tensor = torch.tensor(image_features_selected.squeeze(), dtype=torch.float32)

        return {
            'ct_images': ct_images if isinstance(ct_images, torch.Tensor) else torch.tensor(ct_images,dtype=torch.float32),
            'mask_images': mask_images if isinstance(mask_images, torch.Tensor) else torch.tensor(mask_images,dtype=torch.float32),
            'image_features': image_features_tensor,
            'structured_data': structured_data,
            'label': label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long),
            'patient_name': patient_name
        }
