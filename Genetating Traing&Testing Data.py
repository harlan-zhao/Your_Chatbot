# this file is to get the paired rows in the database we build and save them as input/ouput files

# import dependencies
import sqlite3
import pandas as pd

# connect the database we created from our dataset
connection = sqlite3.connect('new.db')
c = connection.cursor()
limit = 5000
last_unix = 0
cur_length = limit
counter = 0
test_done = True


# get the texts we needed and store them in input/output files
while cur_length == limit:
    df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix, limit), connection)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)
    if not test_done:
        test_done = True
        pass
    else:
        with open("questions", 'a', encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content+'\n')
        with open("answers", 'a', encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(content+'\n')

    counter += 1
    if counter % 2000 == 0:
        print(counter*limit,'rows completed')
