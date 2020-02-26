from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pyknp import Juman
import math

logger = getLogger(__name__)


class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class BertWithJumanModel():
    """学習済みBertを使うやつ Fork:https://github.com/yagays/pytorch_bert_japanese"""
    def __init__(self, bert_path, vocab_file_name="vocab.txt", use_cuda=False):
        self.juman_tokenizer = JumanTokenizer()
        self.model = BertModel.from_pretrained(bert_path)
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name, do_lower_case=False, do_basic_tokenize=False)
        self.use_cuda = use_cuda

    def _preprocess_text(self, text):
        return text.replace(" ", "")

    def get_sentence_embedding(self, text, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):
        preprocessed_text = self._preprocess_text(text)
        n = math.ceil(len(preprocessed_text) / 2048)
        result = [preprocessed_text[idx:idx + n] for idx in range(0, len(preprocessed_text), n)]
        tokens = []
        for t in result:
            tokens += self.juman_tokenizer.tokenize(t)
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens))
        ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])  # max_seq_len-2
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        self.model.eval()
        with torch.no_grad():
            all_encoder_layers, _ = self.model(tokens_tensor)

        embedding = all_encoder_layers[pooling_layer].cpu().numpy()[0]
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        elif pooling_strategy == "CLS_TOKEN":
            return embedding[0]
        else:
            raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")
