import pandas as pd

from sklearn.preprocessing import LabelEncoder
from core.english.constants import CLEAN_TRAIN_DATASET_PATH,\
    CLEAN_VAL_DATASET_PATH, ROUGH_TRAIN_DATASET_PATH, ROUGH_VAL_DATASET_PATH
from utility.textProcessorUtils import convert_emojis, convert_emoticons, remove_emoji, remove_emoticons
from utility.textReplaceUtils import replace_hashtags, replace_user_handles, replace_phone_numbers, replace_urls


def preprocess(text):
    """
    Utility function to preprocess the text.
    @param text: Text sentence to be pre-processed.
    @return: Pre-processed text.
    """
    text = replace_hashtags(text)
    text = replace_user_handles(text)
    text = replace_urls(text)
    text = replace_phone_numbers(text)
    text = convert_emojis(text)
    text = convert_emoticons(text)
    text = remove_emoji(text)
    text = remove_emoticons(text)
    text = ''.join(text.split('\n'))
    text = ' '.join(text.split(' '))
    return text


class EnglishTextPreprocessor:

    def __init__(self, rough_train_file_path, rough_val_file_path, clean_train_file_path, clean_val_file_path):
        """
        Text Preprocessor class for english sentences.
        @param rough_train_file_path: File path for un-processed training dataset.
        @param rough_val_file_path: File path for un-processed validation dataset.
        @param clean_train_file_path: File path for pre-processed training dataset.
        @param clean_val_file_path: File path for pre-processed validation dataset.
        """
        self.rough_train_file_path = rough_train_file_path
        self.rough_val_file_path = rough_val_file_path
        self.clean_train_file_path = clean_train_file_path
        self.clean_val_file_path = clean_val_file_path
        self.label_encoder = LabelEncoder()

    def clean_and_save_dataset(self):
        """
        Cleans and pre-processes the dataset and then saves it on disk.
        """
        self.build_data_fields(self.rough_train_file_path)
        self.build_data_fields(self.rough_val_file_path, is_validation_data=True)

    def save_built_data(self, dataframe, is_validation_data=False):
        """
        Saves the cleaned dataset in a csv file on disk.
        @param dataframe: Dataframe to be saved.
        @param is_validation_data: Flag to check if the dataframe is training or validation.
        """
        if is_validation_data:
            dataframe.to_csv(self.clean_val_file_path, index=False, sep=',')
        else:
            dataframe.to_csv(self.clean_train_file_path, index=False, sep=',')

    def build_data_fields(self, dataset_file_path, is_validation_data=False):
        """
        Reads the un-processed dataset, pre-processes it, converts the string labels to integers.
        @param dataset_file_path: File path for un-processed dataset.
        @param is_validation_data: Flag to check if the dataframe is training or validation.
        """
        dataframe = pd.read_csv(dataset_file_path, sep=',', converters={'tweet': preprocess})
        dataframe['label'] = self.label_encoder.fit_transform(dataframe['label'].values)
        self.save_built_data(dataframe, is_validation_data)


if __name__ == '__main__':
    english_text_preprocessor = EnglishTextPreprocessor(rough_train_file_path=ROUGH_TRAIN_DATASET_PATH,
                                                        rough_val_file_path=ROUGH_VAL_DATASET_PATH,
                                                        clean_train_file_path=CLEAN_TRAIN_DATASET_PATH,
                                                        clean_val_file_path=CLEAN_VAL_DATASET_PATH)
    english_text_preprocessor.clean_and_save_dataset()
