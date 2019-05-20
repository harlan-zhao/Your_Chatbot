from RNN_Model import RNN_Model
from TokenizerWrap import TokenizerWrap
import re

# choose a MODE "train" or "chat"
MODE = "chat"
# MODE = "train"

# build inputs and outputs
data_src, data_dest = [], []
num_words = 10000
filter_amount = 200                     # filter some long sentences that causes wasting computing power
SOS = "ssss "                           # start of sentence mark
EOS = " eeee"                           # end of sentence mark
input_file_path = 'datasets/RC_questions_2' # where you store the questions part(input) of conversations
label_file_path = "datasets/RC_answers_2"   # where you store the answers part(input) of conversations
punc = '[,.!\'?"]'                      # use regex get rid of the punctuations in texts

# open training sets and read them into lists
with open(input_file_path, "rb") as f:
    with open(label_file_path, "rb") as g:
        while True:
            line1 = f.readline()
            line2 = g.readline()
            if not line1 or not line2:
                break
            line1 = str(line1)
            line2 = str(line2)
            if len(line1) > filter_amount or len(line2) > filter_amount:
                continue
            flag = False
            for letter in ["false", "False", "x99d", "newlinechar","xe2", "x80"]:
                if letter in line1 or letter in line2:
                    flag = True
                    break
            if flag:
                continue

            line1 = line1.strip("b\'").replace("\\r\\n", "")
            line1 = re.sub(punc, '', line1)
            line2 = line2.strip("b\'").replace("\\r\\n", "")
            line2 = re.sub(punc, '', line2)
            line2 = SOS + line2 + EOS
            data_src.append(line1)
            data_dest.append(line2)

# tokenize and pad the sequence
source_tokenized = TokenizerWrap(texts=data_src, padding="pre", reverse=True,num_words=num_words)
dest_tokenized = TokenizerWrap(texts=data_dest, padding="post", num_words=num_words)

# use the code below to check whether your data is clean enough by checking the firest 200 most frequent word.
# to see if there are weird word among them.
# res = [(x,y) for x,y in dest_tokenized.index_word.items()]
# res = sorted(res)
# print(res[:200])
# print(len(data_src),1427439)
# print(len(data_dest))

# organize the input config
class input_config(object):
    encoder_input = source_tokenized.padded_tokens
    decoder_input = dest_tokenized.padded_tokens[:, :-1]
    decoder_output = dest_tokenized.padded_tokens[:,1:]  # shift decoder_output one time-step to make it different with the input.
    num_words = num_words                                # num of words in the dictionary you want (not the more the better)
    source_object = source_tokenized
    dest_object = dest_tokenized
    SOS = SOS.strip()
    EOS = EOS.strip()


# organize the model config
class model_config(object):
    embedding_size = 256            # embedding size
    state_size = 512                # state size
    RMSprop_lr = 1e-3               # learning rate for optimizer RMSprop
    ckpt_save_path = "ckpt.keras"   # full path of the checkpoint file
    validation_split = 0.01         # validation split
    batch_size = 1                # batch szie
    epochs = 5                      #epochs


# run the model or chat with the chatbot
model = RNN_Model(input_config(), model_config())
if MODE == "train":
    model.run()
elif MODE == "chat":
    model.run(mode="predict")
else:
    print("Wrong MODE! please choose \"train\" or \"chat\" ")

