# wrap keras tokenizer with some functions we need such as converting int sequence to word seqence

# import libraries
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, padding, max_len, reverse=False, num_words=None):
        '''
        :param texts: text sequence that you want to train with
        :param padding: "pre" for pre-padding, "post" for post-padding
        :param max_len: max length for seqences after padding (in order to save memory)
        :param reverse: if True, the returned int_token sequence would be reversed
        :param num_words: max num of words in the word_token dictionary
        '''
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_word = dict(zip(self.word_index.values(),self.word_index.keys()))
        self.int_seq = self.texts_to_sequences(texts)
        if reverse:
            self.int_seq = [list(reversed(sequence)) for sequence in self.int_seq]

        # length of pre-padding int_sequence
        self.seq_len = [len(seq) for seq in self.int_seq]
        self.max_len = max_len  
        self.padded_tokens = pad_sequences(self.int_seq, maxlen=self.max_len, padding = padding,truncating="post")

    def get_index(self, word):
        # use thi func to get the start mark index and end mark index in the dictionary
        try:
            return self.word_index[word]
        except Exception as e:
            print(e)






