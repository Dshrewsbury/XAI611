from typing import Optional
import os
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from utils.generate_partial_labels import *


# Pascal VOC only has a train/validation datasets
# So we split the train into train/val and use the val dataset as the test set
def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are
    returned because this function is for splitting, not shuffling.
    '''

    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2

    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])

    return (idx_1, idx_2)


class PartialMultiLabel(Dataset):

    def __init__(self,
                 image_names,
                 partial_labels,
                 true_labels,
                 data_dir="",
                 transforms=None):

        self.image_names = image_names
        self.ground_truth_labels = true_labels
        self.data_dir = data_dir
        self.transforms = transforms
        self.labels_pt = partial_labels


    def get_all_images(self):
        images = []
        for i in range(len(self.image_names)):
            image_path = os.path.join(self.data_dir, self.image_names[i])
            image = Image.open(image_path).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
            images.append(image)
        return torch.stack(images)

    def __len__(self):
        return len(self.image_names)

    # returns the now noisy target
    def __getitem__(self, index):
        # Note that LLM returns a feature vector as images as an option, probably for analysis

        image_path = os.path.join(self.data_dir, self.image_names[index])
        with Image.open(image_path) as I_raw:
            image = I_raw.convert('RGB')

        # Where to do this
        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.labels_pt[index, :], self.ground_truth_labels[index, :], index


# ss_seed = seed for subsampling
# ss_frac = fraction of training/val set to subsample
# use_feats is for their linear training
class PartialMLDataModule(LightningDataModule):

    def __init__(
            self,
            batch_size: int = 32,
            num_workers: int = 8,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = 0.2
        self.split_seed = 42
        self.ss_frac_val = 1.0
        self.ss_frac_train = 1.0
        self.ss_seed = 1200

        self.image_dir = './data/VOCdevkit/VOC2012/JPEGImages'
        self.anno_dir = './data/'
        self.save_hyperparameters(logger=False)

        self.transform_train = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load data:
        data = {}
        for phase in ['train', 'val']:
            data[phase] = {}
            data[phase]['labels'] = np.load(os.path.join(self.anno_dir, 'processed_{}_labels.npy'.format(phase)))
            data[phase]['partial_labels'] = np.load(os.path.join(self.anno_dir, 'partial_{}_labels.npy'.format(phase)))
            data[phase]['images'] = np.load(os.path.join(self.anno_dir, 'processed_{}_images.npy'.format(phase)))

        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(data['train']['images']),
            self.val_frac,
            np.random.RandomState(self.split_seed)
        )

        # subsample split indices:
        ss_rng = np.random.RandomState(self.ss_seed)
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])

            # selects a fraction of the validation set/train set, maybe to set fraction of dataset to be sampled
            if phase == 'train':
                num_final = int(np.round(self.ss_frac_train * num_initial))
            else:
                num_final = int(np.round(self.ss_frac_val * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]

        precomputed_partial_labels = ''
        self.data_train = PartialMultiLabel(image_names=data['train']['images'][split_idx['train']],
                                            partial_labels= data['train']['partial_labels'][split_idx['train'], :],
                                            true_labels=data['train']['labels'][split_idx['train'], :],
                                            data_dir=self.image_dir,
                                            transforms=self.transform_train)

        self.data_val = PartialMultiLabel(image_names=data['train']['images'][split_idx['val']],
                                          partial_labels=data['train']['partial_labels'][split_idx['val'], :],
                                          true_labels=data['train']['labels'][split_idx['val'], :],
                                          data_dir=self.image_dir,
                                          transforms=self.transform_val)

        self.data_test = PartialMultiLabel(image_names=data['val']['images'],
                                           partial_labels=data['val']['partial_labels'],
                                           true_labels=data['val']['labels'],
                                           data_dir=self.image_dir,
                                           transforms=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
