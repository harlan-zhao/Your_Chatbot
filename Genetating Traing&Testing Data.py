import sqlite3
import pandas as pd

connection = sqlite3.connect('new.db')
c = connection.cursor()
limit = 5000
last_unix = 0
cur_length = limit
counter = 0
test_done = True

while cur_length == limit:
    df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix, limit), connection)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)
    if not test_done:
        test_done = True
        pass
    else:
        with open("train1", 'a', encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content+'\n')
        with open("train2", 'a', encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(content+'\n')

    counter += 1
    if counter % 2000 == 0:
        print(counter*limit,'rows completed')
