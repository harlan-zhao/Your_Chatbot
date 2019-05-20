# this file is to get the paired rows in the database we build and save them as input/ouput files

# import dependencies
import sqlite3
import pandas as pd
import re

# connect the database we created from our dataset
connection = sqlite3.connect('data.db')
c = connection.cursor()
limit = 5000
last_unix = 0
cur_length = limit
counter = 0
flag = False


# get the texts we needed and store them in input/output files
while True:
    df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix, limit), connection)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)
    punc = '[,.!\'?"]'
    with open("questions", 'a', encoding='utf8') as f:
        with open("answers", 'a', encoding='utf8') as g:
            from_list = df['parent'].values
            to_list = df['comment'].values
            for i in range(len(from_list)):
                line1 = str(from_list[i])
                line2 = str(to_list[i])
                ## you can use the code below to do some cleaning so you don't need to do it in training part
                # if not line1 or not line2:
                #     flag = True
                #     break
                # if len(line1) > 200 or len(line2) > 200:
                #     continue
                # next = False
                # for letter in ["false", "False", "x99d", "newlinechar", "xe2", "x80"]:
                #     if letter in line1 or letter in line2:
                #         next = True
                #         break
                # if next:
                #     continue
                #
                # line1 = line1.strip("b\'").replace("\\r\\n", "")
                # line1 = re.sub(punc, '', line1)
                # line2 = line2.strip("b\'").replace("\\r\\n", "")
                # line2 = re.sub(punc, '', line2)

                f.write(line1 + '\n')
                g.write(line2 + '\n')

    counter += 1
    if counter % 20 == 0:
        print(counter*limit,'rows completed')
    if flag:
        break
