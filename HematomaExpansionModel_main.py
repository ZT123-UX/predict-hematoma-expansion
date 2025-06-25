import joblib
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from models.HematomaExpansionModel import HematomaExpansionModel
from data.dataset_with_dcm import MultimodalDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
from utils import set_seed
import os
import config
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import argparse
import logging
import shap
import cv2
import torch.nn.functional as F



class Trainer:
    def __init__(self, model, num_structured_features, batch_size=config.BATCH_SIZE, learning_rate=1e-4):
        self.model = model
        self.num_structured_features = num_structured_features
        self.batch_size = batch_size
        self.epochs = config.NUM_EPOCHS
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_test_accuracy = 0.0

        self.train_losses = []
        self.test_accuracies = []
        self.test_aucs = []
        self.test_sensitivities = []
        self.test_specificities = []

        self.all_outputs = []
        self.all_targets = []

    def prepare_data(self, dataset):
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)


        print(dataset.data_clean.columns[2:][dataset.selector.get_support()])

        print(dataset.image_feature_data_clean.columns[2:][dataset.image_selector.get_support()])





    def train(self):
        self.train_aucs = []
        self.test_labels = []
        self.test_scores = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for i, data in enumerate(self.train_loader):
                ct_images = data['ct_images'].to(self.device)
                mask_images = data['mask_images'].to(self.device)
                struct_data = data['structured_data'].to(self.device)
                image_features = data['image_features'].to(self.device)
                targets = data['label'].to(self.device)
                patient_name = data['patient_name']
                outputs = self.model(ct_images, mask_images,struct_data,image_features)

                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                targets = targets.float()


                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                preds = (outputs.squeeze() > 0.5).float()
                correct_preds += (preds == targets).sum().item()
                total_preds += targets.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = correct_preds / total_preds
            self.train_losses.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            test_accuracy, auc, specificity, sensitivity, all_targets, all_outputs = self.test()
            self.test_accuracies.append(test_accuracy)
            self.test_aucs.append(auc)
            self.test_sensitivities.append(sensitivity)
            self.test_specificities.append(specificity)
            self.all_outputs.append(all_outputs)
            self.all_targets.append(all_targets)
            print(f"Test Accuracy: {test_accuracy:.4f}, AUC: {auc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")

    def test(self):
        self.model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for data in self.test_loader:
                ct_images = data['ct_images'].to(self.device)
                mask_images = data['mask_images'].to(self.device)
                struct_data = data['structured_data'].to(self.device)
                image_features = data['image_features'].to(self.device)
                targets = data['label'].to(self.device)
                patient_name = data['patient_name']

                outputs = self.model(ct_images, mask_images, struct_data, image_features)
                outputs = outputs.squeeze().float()
                targets = targets.float()

                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        preds = (all_outputs > 0.5).astype(int)

        accuracy = accuracy_score(all_targets, preds)
        auc = roc_auc_score(all_targets, all_outputs)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        return accuracy, auc, specificity, sensitivity, all_targets, all_outputs

    def validate_external_data(self, external_csv_list, external_dcm_dirs, external_mask_dirs, scaler, selector,image_scaler, image_selector,
                               log_path,batch_size=16,epoch=1,):


        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(filename=log_path,
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            filemode='a')

        def log(msg):
            print(msg)
            logging.info(msg)

        external_dataset = MultimodalDataset(
            data_files=external_csv_list,
            image_dirs=external_dcm_dirs,
            mask_dir=external_mask_dirs,
            target_image_count=config.TARGET_IMAGE_COUNT,
            image_feature_list=config.EXTERNAL_IMAGE_FEATURE_PATH,
            selected_feature_count=config.SELECTED_FEATURE_COUNT,
            selected_image_feature_count=config.SELECTED_IMAGE_FEATURE_COUNT,
            transform=None,
            augment=False,
            img_dimensions=config.IMAGE_DIMENSIONS,
            scaler=scaler,
            selector=selector,
            image_scaler=image_scaler,
            image_selector=image_selector
        )

        log("=" * 60)
        log(f"model val - Epoch {epoch}")
        log("=" * 60)

        external_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=False)
        self.train_loader = external_loader

        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        all_labels, all_predictions = [], []
        correct_0, total_0, correct_1, total_1 = 0, 0, 0, 0

        collected_struct_data = []
        collected_outputs = []

        with torch.no_grad():
            for batch in external_loader:
                ct_images = batch['ct_images'].to(device)
                mask_images = batch['mask_images'].to(device)
                struct_data = batch['structured_data'].to(device)
                image_features = batch['image_features'].to(device)
                labels = batch['label'].to(device)
                patient_name = batch['patient_name']

                outputs = self.model(ct_images, mask_images, struct_data, image_features)
                predicted = (outputs.squeeze() > 0.5).float()


                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.squeeze().cpu().numpy())

                print(struct_data.shape)
                print(outputs.shape)
                collected_struct_data.append(struct_data.cpu())
                collected_outputs.append(outputs.cpu())

                for i in range(len(labels)):
                    if labels[i].item() == 0:
                        total_0 += 1
                        correct_0 += (predicted[i].item() == 0)
                    elif labels[i].item() == 1:
                        total_1 += 1
                        correct_1 += (predicted[i].item() == 1)
                    log(f"patient_name: {patient_name[i]}, true_label: {labels[i].item()}, predict_label: {outputs[i].item():.4f}")

        external_accuracy = 100 * (correct_0 + correct_1) / (total_0 + total_1)
        external_auc = roc_auc_score(all_labels, all_predictions)

        accuracy_0 = 100 * correct_0 / total_0 if total_0 > 0 else 0
        accuracy_1 = 100 * correct_1 / total_1 if total_1 > 0 else 0

        tn, fp, fn, tp = confusion_matrix(all_labels, np.round(all_predictions)).ravel()
        external_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        external_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        log(f"ACC: {external_accuracy:.2f}%, AUC: {external_auc:.4f}, "
              f"Sensitivity: {external_sensitivity:.4f}, Specificity: {external_specificity:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hematoma Expansion Model Training and Testing")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Choose 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == "train":
        os.environ['OMP_NUM_THREADS'] = '1'

        set_seed(seed=42)

        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)


        train_dataset = MultimodalDataset(
            data_files=config.CSV_FILE_PATH,
            image_dirs=config.DCM_DIR_PATH,
            mask_dir=config.IMAGE_DIR_PATH,
            target_image_count=config.TARGET_IMAGE_COUNT,
            img_dimensions=(256, 256),
            # augment=config.USE_AUGMENTATION,
            selected_feature_count=config.SELECTED_FEATURE_COUNT,
            image_feature_list=config.IMAGE_FEATURE_PATH,
            # selected_image_feature_count=config.SELECTED_IMAGE_FEATURE_COUNT,
        )
        print(train_dataset)

        joblib.dump(train_dataset.scaler, 'HEM_scaler.pkl')
        joblib.dump(train_dataset.selector, 'HEM_selector.pkl')

        joblib.dump(train_dataset.image_scaler, 'HEM_image_feature_scaler.pkl')
        joblib.dump(train_dataset.image_selector, 'HEM_image_feature_selector.pkl')

        model = HematomaExpansionModel(struct_feat_dim=config.SELECTED_FEATURE_COUNT,image_feat_dim=config.SELECTED_IMAGE_FEATURE_COUNT)

        trainer = Trainer(model, num_structured_features=config.SELECTED_FEATURE_COUNT, batch_size=config.BATCH_SIZE)

        trainer.prepare_data(train_dataset)

        print("train start")
        trainer.train()
    elif args.mode == "test":
        set_seed(seed=42)

        scaler = joblib.load('HEM_scaler.pkl')
        selector = joblib.load('HEM_selector.pkl')

        image_scaler = joblib.load('HEM_image_feature_scaler.pkl')
        image_selector = joblib.load('HEM_image_feature_selector.pkl')

        external_csv = config.EXTERNAL_CSV
        external_dcm_dirs = []
        external_mask_dirs = []

        model_dir = "./saved_models"
        for epoch in range(config.MODEL_START, config.MODEL_END):
            model_path = os.path.join(model_dir, f"{config.MODEL_NAME}_HEM_model_epoch_{epoch}.pth")
            if not os.path.exists(model_path):
                continue

            model = HematomaExpansionModel(struct_feat_dim=config.SELECTED_FEATURE_COUNT,image_feat_dim=config.SELECTED_IMAGE_FEATURE_COUNT)
            trainer = Trainer(model, num_structured_features=config.SELECTED_FEATURE_COUNT)

            trainer.model.load_state_dict(torch.load(model_path), strict=False)

            trainer.validate_external_data(external_csv_list=external_csv,
                                           external_dcm_dirs=external_dcm_dirs,
                                           external_mask_dirs=external_mask_dirs,
                                           scaler=scaler,
                                           selector=selector, epoch=epoch,
                                           image_scaler=image_scaler,
                                           image_selector=image_selector,
                                           log_path=config.LOG_PATH)





