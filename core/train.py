import torch

from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from core.english.constants import TRAIN_EPOCHS, TRAIN_BATCH_SIZE, \
    EVAL_BATCH_SIZE, CLEAN_TRAIN_DATASET_PATH, CLEAN_VAL_DATASET_PATH, LEARNING_RATE, MODEL_FILE_PATH
from core.english.model import RobertaForBinaryClassification
from core.english.text_dataset import TextDataset


def get_data():
    train_dataset = TextDataset(dataset_file_path=CLEAN_TRAIN_DATASET_PATH)
    val_dataset = TextDataset(dataset_file_path=CLEAN_VAL_DATASET_PATH)

    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)

    return train_dl, val_dl


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    print("Retrieving the data loaders used to train the model...")
    train_dl, val_dl = get_data()

    print("Building the model...")

    num_labels = 1

    model_config = AutoModel.from_pretrained("roberta-base").config
    model = RobertaForBinaryClassification(model_config, num_labels, device)
    # model.load_state_dict(torch.load(MODEL_FILE_PATH))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    global_val_score = -1000

    print("Stared training the model...")

    for epoch in range(TRAIN_EPOCHS):
        print("Epoch {}/{}".format(epoch, TRAIN_EPOCHS - 1))
        print("-" * 100)

        model.train()

        running_loss = 0.0
        train_tqdm_iterator = tqdm(train_dl, total=int(len(train_dl)))
        counter = 0

        for _, batch in enumerate(train_tqdm_iterator):
            inputs = list(batch[1])
            true_labels = batch[2]

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                loss, logits = model.forward(inputs, true_labels, device)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * len(batch[0])
            counter += 1
            train_tqdm_iterator.set_postfix(loss=(running_loss / (counter * train_dl.batch_size)))

        epoch_loss = running_loss / len(train_dl)
        print("Training Loss: {:.4f}".format(epoch_loss))

        model.eval()
        with torch.no_grad():
            fin_targets = []
            fin_outputs = []
            val_tqdm_iterator = tqdm(val_dl, total=int(len(val_dl)))

            for _, batch in enumerate(val_tqdm_iterator):
                inputs = list(batch[1])
                yb = batch[2]
                loss, logits = model.forward(inputs, yb, device)
                yb = yb.cpu().detach().numpy().tolist()
                out = torch.sigmoid(logits).cpu().detach().numpy().tolist()
                fin_targets.extend(yb)
                fin_outputs.extend(out)

            outputs = []
            for i in range(len(fin_outputs)):
                if fin_outputs[i][0] >= 0.5:
                    outputs.append(1.0)
                else:
                    outputs.append(0.0)
            val_score = metrics.f1_score(fin_targets, outputs)

        if val_score > global_val_score:
            global_val_score = val_score
            torch.save(model.state_dict(), MODEL_FILE_PATH)

        print("The F1 score for epoch ", epoch, " is ", val_score)
        scheduler.step(1 - val_score)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    train()
