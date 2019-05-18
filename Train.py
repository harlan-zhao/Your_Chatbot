from RNN_Model import RNN_Model
from TokenizerWrap import TokenizerWrap
import re

# build inputs and outputs
data_src, data_dest = [], []
SOS = "SSSS "   # start of sentence mark
EOS = " EEEE"   # end of sentence mark
input_file_path = 'datasets/questions'
label_file_path = "datasets/answers"
punc = '[,.!\'?"]'             # use regex get rid of the punctuations in texts

with open(input_file_path, "rb") as f:
    with open(label_file_path, "rb") as g:
        while True:
            line1 = f.readline()
            line2 = g.readline()
            if not line1 or not line2:
                break
            line1 = str(line1)
            line2 = str(line2)
            if len(line1) > 150 or len(line2) > 150:
                continue
            flag = False
            for letter in ["xe2", "x80", "x99d"]:
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
print(data_dest[1])
print(data_src[1])