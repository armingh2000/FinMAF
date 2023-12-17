from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA


def get_embedding_input(metadata):
    inputs = {}

    for _, row in tqdm(metadata.iterrows()):
        inputs[row["Symbol"]] = row["Security Name"]

    return inputs


def reduce_dimensions(embeddings, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings


class EmbeddingModel:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.model, self.tokenizer = self.load_embedding_model(model_name)

    def load_embedding_model(self, model_name):
        # Load pre-trained model and tokenizer
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def __call__(self, symbol, description):
        with torch.no_grad():
            # Tokenize the stock symbols with their descriptions and get embeddings
            # The tokenizer automatically adds [CLS] and [SEP] tokens
            inputs = self.tokenizer(
                symbol, description, return_tensors="pt", padding=True, truncation=True
            )
            output = self.model(**inputs)
            cls_hidden_state = output.last_hidden_state[:, 0, :]

            return cls_hidden_state


def get_embeddings(embedding_inputs):
    embeddings = {}
    embedding_model = EmbeddingModel()

    for symbol, description in tqdm(embedding_inputs.items()):
        embeddings[symbol] = embedding_model(symbol, description)

    return embeddings
