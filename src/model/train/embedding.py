from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import src.configs as configs


def get_embedding_input(metadata, logger):
    logger.info("Getting embedding dictionary for BERT input ...")

    inputs = {}

    for _, row in tqdm(metadata.iterrows()):
        inputs[row["Symbol"]] = row["Security Name"]

    return inputs


def perform_pca(embeddings, logger):
    logger.info("Preparing embeddings for PCA ...")
    stocks = list(embeddings.keys())
    embeddings = [embeddings[stock] for stock in stocks]
    embeddings = np.array(embeddings).squeeze()

    logger.info("Performing PCA ...")
    pca = PCA(n_components=configs.n_pca_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    logger.info("Generating embedding dictionary ...")
    result = {stocks[i]: np.array(reduced_embeddings[i]) for i in range(len(stocks))}

    return result


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


def get_embeddings(embedding_inputs, logger):
    embeddings = {}

    logger.info("Loading BERT model for embeddings ...")
    embedding_model = EmbeddingModel()

    logger.info("Getting embeddings ...")
    for symbol, description in tqdm(embedding_inputs.items()):
        embeddings[symbol] = embedding_model(symbol, description)

    return embeddings
