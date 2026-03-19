import os
import sys
import yaml
import shutil
import zipfile
import subprocess
from WasteDetection.utils.main_utils import read_yaml_file
from WasteDetection.logger import logging
from WasteDetection.entity.config_entity import ModelTrainerConfig
from WasteDetection.entity.artifacts_entity import ModelTrainerArtifact
from WasteDetection.exception import AppException


class ModelTrainer:
    def __init__(
        self, 
        model_trainer_config: ModelTrainerConfig
    ):
        self.model_trainer_config = model_trainer_config
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class")
        
        try:
            logging.info("Unzipping data")
            # Use cross-platform Python commands
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("data.zip")

            # Fix data.yaml paths for Windows compatibility
            with open("data.yaml", 'r') as stream:
                data_config = yaml.safe_load(stream)
            
            num_classes = str(data_config['nc'])
            
            # Update paths to be relative to project root (where data.yaml is located)
            data_config['train'] = 'train/images'
            data_config['val'] = 'valid/images'
            
            with open("data.yaml", 'w') as f:
                yaml.dump(data_config, f)

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)        
            
            # Use subprocess with cwd for cross-platform compatibility
            result = subprocess.run(
                [
                    sys.executable, 'train.py',
                    '--img', '416',
                    '--batch', str(self.model_trainer_config.batch_size),
                    '--epochs', str(self.model_trainer_config.no_epochs),
                    '--data', '../data.yaml',
                    '--cfg', './models/custom_yolov5s.yaml',
                    '--weights', self.model_trainer_config.weight_name,
                    '--name', 'yolov5s_results',
                    '--cache'
                ],
                cwd='yolov5',
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logging.error(f"Training stdout: {result.stdout}")
                logging.error(f"Training stderr: {result.stderr}")
                raise Exception(f"Training failed: {result.stderr}")
            
            # Use cross-platform Python commands instead of Unix commands
            best_model_path = os.path.join('yolov5', 'runs', 'train', 'yolov5s_results', 'weights', 'best.pt')
            
            # Copy to yolov5 directory
            shutil.copy(best_model_path, 'yolov5')
            
            # Copy to model trainer directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(best_model_path, self.model_trainer_config.model_trainer_dir)
            
            # Cleanup using cross-platform commands
            if os.path.exists('yolov5/runs'):
                shutil.rmtree('yolov5/runs')
            if os.path.exists('train'):
                shutil.rmtree('train')
            if os.path.exists('valid'):
                shutil.rmtree('valid')
            if os.path.exists('data.yaml'):
                os.remove('data.yaml')
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = 'yolov5/best.pt'
                
            )
            
            logging.info("Exited the initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)
            