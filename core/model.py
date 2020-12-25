import torch

from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, RobertaModel, AlbertPreTrainedModel


class RobertaForBinaryClassification(AlbertPreTrainedModel):

    def __init__(self, model_config, num_labels, device):
        super(RobertaForBinaryClassification, self).__init__(model_config)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_config.num_labels = num_labels
        self.num_labels = num_labels
        self.model = RobertaModel.from_pretrained("roberta-base", output_attentions=True).to(device)
        self.dropout = torch.nn.Dropout(model_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(model_config.hidden_size, num_labels)
        self.loss_fn = BCEWithLogitsLoss()
        self.apply(self._init_weights)

    def forward(self, text_batch, labels, device, is_visualization=False):
        if labels is not None:
            labels = labels.to(device)

        if not is_visualization:
            inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            model_inputs = {}
            for key, val in inputs.items():
                model_inputs[key] = val.to(device)
        else:
          inputs = text_batch

        inputs = inputs.to(device)
        attention = None
        if is_visualization:
            outputs = self.model(inputs)
            attention = outputs[-1]
        else:
            outputs = self.model(**model_inputs)
        
        output_1 = outputs[0][:, 0]
        output_2 = self.dropout(output_1)
        logits = self.classifier(output_2)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return None, logits, attention
