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


class PyviTokenizer(Tokenizer, Component):
    name = "tokenizer_pyvi"

    provides = ["tokens"]

    language_list = ["vi"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pyvi"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        from pyvi.pyvi import ViTokenizer
        from underthesea import word_sent
        # type: (Text) -> List[Token] 
        # Vietnamese pyvi
        #tokenizer = ViTokenizer()
        #words = tokenizer.tokenize(text)

        # Vietnamese underthesea
        words = word_sent(text)           
        tokenized = [(word, text.find(word), text.find(word) + len(word)) for word in words]
        tokens = [Token(word, start) for (word, start, end) in tokenized]        
   
        return tokens

