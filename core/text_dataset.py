import torch
import pandas as pd

from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, dataset_file_path, is_test=False):
        """
        This class is used to create the torch dataset for the sentences from the file path.
        @param dataset_file_path: File path for the dataset stored on disk.
        @param is_test: Flag to check is the dataset is test or not.
        """
        self.dataset_file_path = dataset_file_path
        self.is_test = is_test

        dataframe = pd.read_csv(self.dataset_file_path, sep=',')

        self.ids = dataframe['id'].values
        self.tweets = dataframe['tweet'].values

        if not is_test:
            self.labels = dataframe['label'].values

    def __len__(self):
        """
        @return: Length of the current batch of sentences.
        """
        return self.tweets.shape[0]

    def __getitem__(self, index):
        """
        Gets the sentences and labels from the torch dataset.
        @param index: Index of the dataset used to fetch elements.
        @return: Sentences and labels from the torch dataset.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        if self.is_test:
            return self.ids[index], self.tweets[index]
        else:
            return self.ids[index], self.tweets[index], torch.tensor(self.labels[index], dtype=torch.float32)
