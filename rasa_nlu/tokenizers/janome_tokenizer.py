from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import logging

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Text

logger = logging.getLogger(__name__)


class JanomeTokenizer(Tokenizer, Component):
    name = "tokenizer_janome"

    provides = ["tokens"]

    language_list = ["ja"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["janome"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        from janome.tokenizer import Tokenizer
        import tinysegmenter
        import MeCab
        # type: (Text) -> List[Token]
        
        # Japanese tinysegmenter
        #tokenizer = tinysegmenter.TinySegmenter()
        #words = tokenizer.tokenize(text)
   
        # Japanese janome
        #tokenizer = Tokenizer()
        #words = tokenizer.tokenize(text, wakati = "True")
        #tokenized = [(word, text.index(word), text.index(word) + len(word)) for word in words]

        # Japanese Mecab
        m = MeCab.Tagger(" -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")
        m.parse("")
        node = m.parseToNode(text)
        words = []
        while node:
             words.append(node.surface)
             node = node.next

        tokenized = [(word, text.find(word), text.find(word) + len(word)) for word in words]
        tokens = [Token(word, start) for (word, start, end) in tokenized]
        
        return tokens

