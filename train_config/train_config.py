import os
from abc import ABCMeta, abstractmethod

from pietoolbelt.models.decoders.unet import UNetDecoder
from torch.nn import Module
from torch.optim import Adam

from piepline.data_producer import DataProducer
from piepline.train_config.stages import TrainStage, ValidationStage
from piepline.train_config.train_config import BaseTrainConfig
from pietoolbelt.datasets.utils import DatasetsContainer
from pietoolbelt.losses.regression import RMSELoss
from pietoolbelt.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34, ClassificationModel

from train_config.config import batch_size
from train_config.dataset import create_augmented_dataset
from config import INDICES_DIR, TRAIN_DIR

__all__ = ['TrainConfig', 'ResNet18TrainConfig', 'ResNet34TrainConfig']


class TrainConfig(BaseTrainConfig, metaclass=ABCMeta):
    def __init__(self, fold_indices: {}):
        model = self.create_model(pretrained=False).cuda()

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset(is_train=True, indices_path=os.path.join(INDICES_DIR, indices + '.npy')))

        val_dts = create_augmented_dataset(is_train=False, indices_path=os.path.join(INDICES_DIR, fold_indices['val'] + '.npy'))

        workers_num = 4
        self._train_data_producer = DataProducer(DatasetsContainer(train_dts), batch_size=batch_size, num_workers=workers_num). \
            global_shuffle(True)#.pin_memory(True)
        self._val_data_producer = DataProducer(val_dts, batch_size=batch_size, num_workers=workers_num). \
            global_shuffle(True)#.pin_memory(True)

        self.train_stage = TrainStage(self._train_data_producer)
        self.val_stage = ValidationStage(self._val_data_producer)

        loss = RMSELoss().cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model(pretrained: bool = True) -> Module:
        pass

    @staticmethod
    def _create_decoder(encoder: Module) -> Module:
        model = ClassificationModel(encoder, in_features=512, classes_num=2)
        return ModelWithActivation(model, activation='sigmoid')


class ResNet18TrainConfig(TrainConfig):
    model_name = 'resnet18'
    experiment_dir = os.path.join(TRAIN_DIR, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=3)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet')
        return TrainConfig._create_decoder(enc)


class ResNet34TrainConfig(TrainConfig):
    model_name = 'resnet34'
    experiment_dir = os.path.join(TRAIN_DIR, model_name)

    @staticmethod
    def create_model(pretrained: bool = True) -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=3)
        if pretrained:
            ModelsWeightsStorage().load(enc, 'imagenet')
        return TrainConfig._create_decoder(enc)
