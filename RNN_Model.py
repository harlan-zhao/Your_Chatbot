# import dependencies
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
class RNN_Model(object):
    def __init__(self, input_config, model_config):
        '''
        :param config: pass all  the configurations the model needs
        '''

        self.encoder_input_data = input_config.encoder_input
        self.decoder_input_data = input_config.decoder_input
        self.decoder_output_data = input_config.decoder_output
        self.num_words = input_config.num_words
        self.source_object = input_config.source_object
        self.dest_object = input_config.dest_object
        self.SOS = input_config.SOS
        self.EOS = input_config.EOS
        self.embedding_size = model_config.embedding_size      # embedding size
        self.state_size = model_config.state_size               # state size
        self.lr = model_config.RMSprop_lr                     # learning rate for optimizer RMSprop
        self.save_path = model_config.ckpt_save_path           # full path of the checkpoint file
        self.validation_split = model_config.validation_split  # validation split
        self.batch_size = model_config.batch_size
        self.epochs = model_config.epochs

    def run(self, mode=None):

        # build the encoder, define the placeholders
        encoder_input = Input(shape=(None,), name='encoder_input')
        encoder_embedding = Embedding(input_dim=self.num_words, output_dim=self.embedding_size, name='encoder_embedding')
        encoder_gru1 = GRU(self.state_size, name='encoder_gru1', return_sequences=True)
        encoder_gru2 = GRU(self.state_size, name='encoder_gru2', return_sequences=True)
        encoder_gru3 = GRU(self.state_size, name='encoder_gru3', return_sequences=False)

        # connect all the placeholders in encoder layer
        def encoder_builder():
            encoder_link = encoder_input
            encoder_link = encoder_embedding(encoder_link)
            encoder_link = encoder_gru1(encoder_link)
            encoder_link = encoder_gru2(encoder_link)
            encoder_link = encoder_gru3(encoder_link)
            encoder_output1 = encoder_link
            return encoder_output1
        encoder_output = encoder_builder()

        # build the decoder, define the placeholders
        decoder_initial_state = Input(shape=(self.state_size,),name='decoder_initial_state')
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(input_dim=self.num_words, output_dim=self.embedding_size, name='decoder_embedding')
        decoder_gru1 = GRU(self.state_size, name='decoder_gru1', return_sequences=True)
        decoder_gru2 = GRU(self.state_size, name='decoder_gru2', return_sequences=True)
        decoder_gru3 = GRU(self.state_size, name='decoder_gru3', return_sequences=True)
        decoder_dense = Dense(self.num_words, activation='linear', name='decoder_output')

        # connect all the placeholders in decoder layerd
        def decoder_builder(initial_state):
            decoder_link = decoder_input
            decoder_link = decoder_embedding(decoder_link)
            decoder_link = decoder_gru1(decoder_link, initial_state=initial_state)
            decoder_link = decoder_gru2(decoder_link, initial_state=initial_state)
            decoder_link = decoder_gru3(decoder_link, initial_state=initial_state)
            decoder_output1 = decoder_dense(decoder_link)
            return decoder_output1


        # adjusted loss function, cuz the original sparse_crossentropy gives an error for some reasons
        def sparse_crossentropy(y_true, y_pred):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            loss_mean = tf.reduce_mean(loss)
            return loss_mean

        # connect the model
        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_builder(initial_state=encoder_output)])
        model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])
        decoder_output = decoder_builder(initial_state=decoder_initial_state)
        model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])

        # finalized the model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        # optimizer = RMSprop(lr=self.lr)
        optimizer = RMSprop(lr=self.lr)
        model.compile(optimizer=optimizer, loss=sparse_crossentropy, target_tensors=[decoder_target])

        # ser early-stopping and tensorboard and save check point
        checkpoint = ModelCheckpoint(filepath=self.save_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
        callbacks = [early_stopping, checkpoint, tensorboard]

        #load check point if exists
        try:
            model.load_weights(self.save_path)
            print("CheckPoint Loaded,{}".format(self.save_path))
        except Exception as error:
            print("Failed CheckPoint")
            print(error)

        # pack inputs and outputs
        input_data = {'encoder_input': self.encoder_input_data, 'decoder_input': self.decoder_input_data}
        output_data = {'decoder_output': self.decoder_output_data}

        # start training
        if mode == None:
            model.fit(x=input_data, y=output_data, batch_size=self.batch_size, epochs=self.epochs,
                      validation_split=self.validation_split, callbacks=callbacks)

        # make predictions for interference
        elif mode == "predict":
            start_index = self.dest_object.get_index(self.SOS)
            end_index = self.dest_object.get_index(self.EOS)
            def predict(text):
                input_tokens = self.source_object.text_to_tokens(text=text, reverse=True, padding=True)
                initial_state = model_encoder.predict(input_tokens)
                max_length = self.dest_object.max_len
                decoder_input_data = np.zeros(shape=(1, max_length), dtype=np.int)
                token_int = start_index
                output_text = ""
                count = 0
                while token_int != end_index and count < max_length:
                    decoder_input_data[0, count] = token_int
                    x_data = {'decoder_initial_state': initial_state, 'decoder_input': decoder_input_data}

                    # Input this data to the decoder and get the predicted output.
                    decoder_output = model_decoder.predict(x_data)
                    # Get the last predicted token as a one-hot encoded array.
                    token_onehot = decoder_output[0, count, :]

                    # Convert to an integer-token.
                    token_int = np.argmax(token_onehot)
                    # Lookup the word corresponding to this integer-token.
                    sampled_word = self.dest_object.token_word(token_int)
                    if sampled_word != self.EOS:
                        # Append the word to the output-text.
                        output_text += " " + sampled_word

                    # Increment the token-counter.
                    count += 1
                # add some basic error answers
                if output_text == "":
                    answers = ["i don't know what to say",
                               "i not not telling you what i am thinking",
                               "i don't understand a single word you said",
                               "am i stupied? cuz i don't know what you are talking about",
                               "pardon?",
                               "sorry, i am gonna work harder to understand you",
                               "that is something i don't know what to say"]
                    random = np.random.randint(len(answers))
                    output_text = answers[random]
                return output_text

            # GUI (if MODE ="chat", interact with GUI, go testing it out)
            import tkinter as tk
            window = tk.Tk()
            window.title("Smart Chatbot")
            # window.configure(background="gray")
            window.geometry("850x600")
            display = tk.Text(window, height=12, width=30, font=("Helvetica", 28))
            display.grid(column=0, row=1)

            def output():
                res = str(entry.get())
                entry.delete(0, "end")
                display.insert(tk.END, "You: " + res + "\n")
                answer = predict(res)
                display.insert(tk.END, "ChatBot" + answer + "\n")
                display.see("end")

            title = tk.Label(text="Chatbot")
            title.grid(column=0, row=0)

            img = tk.PhotoImage(file="chatbots.gif")
            lable = tk.Label(window, image=img)
            lable.grid(row=1, column=1)

            entry = tk.Entry(window, width=100)
            entry.grid(column=0, row=2)

            button1 = tk.Button(text="send", width=20, command=output)
            button1.grid(column=1, row=2)

            scrollb = tk.Scrollbar(window, command=display.yview)
            scrollb.grid(row=1, column=0, sticky='nsew')
            display['yscrollcommand'] = scrollb.set

            window.mainloop()






