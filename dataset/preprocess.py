import os
import re
import pandas as pd
import pydicom
import numpy as np


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_dicom_images(patient_folder):
    dicom_files = []
    for root, _, files in os.walk(patient_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return sorted(dicom_files, key=lambda x: natural_sort_key(os.path.basename(x)))


def apply_mask_to_dcm(dcm_file, mask_file, output_dir):
    try:
        dcm = pydicom.dcmread(dcm_file)
        mask = pydicom.dcmread(mask_file)

        if dcm.file_meta.TransferSyntaxUID.is_compressed:
            dcm.decompress()

        if mask.file_meta.TransferSyntaxUID.is_compressed:
            mask.decompress()

        dcm_array = dcm.pixel_array
        mask_array = mask.pixel_array
        if dcm_array.shape != mask_array.shape:
            raise ValueError(f"Shape mismatch: DCM {dcm_array.shape} vs MASK {mask_array.shape}")

        masked_array = np.where(mask_array > 0, dcm_array, 0)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(dcm_file))
        dcm.PixelData = masked_array.tobytes()
        dcm.save_as(output_file)
    except Exception as e:
        print(f"Error processing {dcm_file} with {mask_file}: {e}")


def process_patients(csv_path, dcm_dirs, mask_dirs, output_base_dir):
    if not os.path.exists(csv_path):
        return

    data = pd.read_csv(csv_path)
    for name in data['patient_name']:

        dcm_files = []
        for dcm_dir in dcm_dirs:
            patient_folder = os.path.join(dcm_dir, name)
            if os.path.exists(patient_folder):
                dcm_files.extend(load_dicom_images(patient_folder))


        mask_files = []
        for mask_dir in mask_dirs:
            patient_folder = os.path.join(mask_dir, name)
            if os.path.exists(patient_folder):
                mask_files.extend(load_dicom_images(patient_folder))

        if len(dcm_files) != len(mask_files):
            print(f"File count mismatch for {name}: {len(dcm_files)} DCM vs {len(mask_files)} MASK")
            continue

        output_dir = os.path.join(output_base_dir, name)
        for dcm_file, mask_file in zip(dcm_files, mask_files):
            apply_mask_to_dcm(dcm_file, mask_file, output_dir)


if __name__ == "__main__":

    # data
    csv_path = ''
    dcm_dirs = []
    mask_dirs = []
    output_base_dir = ''

    process_patients(csv_path, dcm_dirs, mask_dirs, output_base_dir)
