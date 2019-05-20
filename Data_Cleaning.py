# The datasets for this project is based on Reddit Comments.
# You can use the link below to get the compressed file for Reddit Comments of every month.(enormous size)
# link----   http://files.pushshift.io/reddit/comments/

# import dependencies
import sqlite3
import json
from datetime import datetime

# build sqlite database and define some parameters for later use
sql_transaction = []
start_row = 0
cleanup = 1000000
data_dir = "G:\AI\Chatbot datasets\RC"

connection = sqlite3.connect('{}.db'.format("data"))
c = connection.cursor()


# operation with database to initialize a table with rows we need
def create_table():
    c.execute(
        "CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, unix INT, score INT)")


# get rid of the newlinechars in dataset
def format_data(data):
    data = data.replace('\n', '').replace('\r', '').replace('"', "'")
    return data


# it is better to commit to database with a trasaction list witch could save a lot of time
def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


# to find if a comment have a parent in the database we are buiding, if so, pair them together
# then we use those pairs to train our RNN
def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        print(str(e))
        return False


# database insert function to replace a row with new values that has a higher score(more accurate answers)
def sql_replace_comment(commentid, parentid, parent, comment, time, score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, unix = ?, score = ? WHERE parent_id =?;""".format(
            parentid, commentid, parent, comment, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# if a line has parent in our table, pair them and insert into table
def sql_has_parent(commentid, parentid, parent, comment, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(
            parentid, commentid, parent, comment, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# if a line is new(does not have parent in the database), insert it into the table
def sql_no_parent(commentid, parentid, comment, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, unix, score) VALUES ("{}","{}","{}",{},{});""".format(
            parentid, commentid, comment, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# set some rules to filter out some unuseful data
def acceptable(data):
    if len(data.split(' ')) > 1000 or len(data) < 1:
        return False
    elif len(data) > 32000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


# find the current score of this parent comment's current existing comment(replace it if its higher)
def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        # print(str(e))
        return False


# run the script
if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
    # data_dir = "G:\AI\Chatbot datasets\RC"   # change the path to where you store your source files
    with open(data_dir, buffering=1000) as f:
        for row in f:
            row_counter += 1
            if row_counter > start_row:
                try:
                    row = json.loads(row)
                    parent_id = row['parent_id']
                    body = format_data(row['body'])
                    created_utc = row['created_utc']
                    score = row['score']
                    comment_id = "t1_" + row['id']

                    parent_data = find_parent(parent_id)
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(body):
                                sql_replace_comment(comment_id, parent_id, parent_data, body, created_utc, score)

                    else:
                        if acceptable(body):
                            if parent_data:
                                if score >= 2:
                                    sql_has_parent(comment_id, parent_id, parent_data, body, created_utc, score)
                                    paired_rows += 1
                            else:
                                sql_no_parent(comment_id, parent_id, body, created_utc, score)
                except Exception as e:
                    print(str(e))

            if row_counter % 100000 == 0:   # print how many lines have been processed
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows,
                                                                              str(datetime.now())))

            if row_counter > start_row:     # delete unpaired rows to make the database neat
                if row_counter % cleanup == 0:
                    print("Cleanin up!")
                    sql = "DELETE FROM parent_reply WHERE parent IS NULL"
                    c.execute(sql)
                    connection.commit()
                    c.execute("VACUUM")
                    connection.commit()