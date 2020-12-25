import torch
import collections
import pandas as pd

from transformers import AutoModel
from torch.utils.data import DataLoader
from core.english.EnglishTextPreprocessor import preprocess
from core.english.constants import EVAL_BATCH_SIZE, MODEL_FILE_PATH, OUTPUT_ANSWER_FILE_PATH, CLEAN_VAL_DATASET_PATH
from core.english.model import RobertaForBinaryClassification
from core.english.text_dataset import TextDataset


class TestPredictor:

    def __init__(self, binary_model_path, rough_test_dataset_path, clean_test_dataset_path, output_text_file):
        self.binary_model_path = binary_model_path
        self.rough_test_dataset_path = rough_test_dataset_path
        self.clean_test_dataset_path = clean_test_dataset_path
        self.output_text_file = output_text_file
        self.predictions = {}

    def predict_answers(self):
        if self.rough_test_dataset_path:
            self.clean_test_dataset()
        self.predict_binary_classes()
        self.save_answer_file()

    def clean_test_dataset(self):
        dataframe = pd.read_csv(self.rough_test_dataset_path, sep=',', converters={'tweet': preprocess})
        dataframe.to_csv(self.clean_test_dataset_path, sep=',', index=False)

    def predict_binary_classes(self):
        test_dataset = TextDataset(dataset_file_path=self.clean_test_dataset_path, is_test=True)
        test_dl = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        print("Retrieving the saved model...")
        num_labels = 1

        model_config = AutoModel.from_pretrained("roberta-base").config
        model = RobertaForBinaryClassification(model_config, num_labels, device)
        model.load_state_dict(torch.load(self.binary_model_path))
        model = model.to(device)

        model.eval()
        with torch.no_grad():
            tweet_ids = []
            fin_outputs = []

            for tweet_id, xb in test_dl:
                tweet_id = list(tweet_id)
                inputs = list(xb)
                loss, logits = model.forward(inputs, None, device)
                out = torch.sigmoid(logits).cpu().detach().numpy().tolist()
                tweet_ids.extend(tweet_id)
                fin_outputs.extend(out)

            for i in range(len(fin_outputs)):
                if fin_outputs[i][0] >= 0.5:
                    self.predictions[tweet_ids[i].item()] = "real"
                else:
                    self.predictions[tweet_ids[i].item()] = "fake"

    def save_answer_file(self):
        ids = []
        labels = []
        pred_dict = collections.OrderedDict(sorted(self.predictions.items()))
        for k, v in pred_dict.items():
            ids.append(k)
            labels.append(v)

        dataframe = pd.DataFrame({
            'id': ids,
            'label': labels
        })
        dataframe.to_csv(self.output_text_file, sep=',', index=False)


if __name__ == '__main__':
    english_test_predictor = TestPredictor(binary_model_path=MODEL_FILE_PATH,
                                           rough_test_dataset_path=None,
                                           clean_test_dataset_path=CLEAN_VAL_DATASET_PATH,
                                           output_text_file=OUTPUT_ANSWER_FILE_PATH)
    english_test_predictor.predict_answers()
