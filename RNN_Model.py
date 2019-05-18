# import dependencies
import tensorflow as tf
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
        self.embedding_size = model_config.embedding_size      # embedding size
        self.state_size = model_config.state_size               # state size
        self.lr =  model_config.RMSprop_lr                     # learning rate for optimizer RMSprop
        self.save_path = model_config.ckpt_save_path           # full path of the checkpoint file
        self.validation_split = model_config.validation.split  # validation split

    def run(self):
        # build the encoder, define the placeholders
        encoder_input = Input(shape=(None,), name='encoder_input')
        encoder_embedding = Embedding(input_dim=self.num_words, output_dim=self.embedding_size, name='encoder_embedding')
        encoder_gru1 = GRU(self.state_size, name='encoder_gru1',
                           return_sequences=True)
        encoder_gru2 = GRU(self.state_size, name='encoder_gru2',
                           return_sequences=True)
        encoder_gru3 = GRU(self.state_size, name='encoder_gru3',
                           return_sequences=False)

        # connect all the placeholders in encoder layer
        encoder_link = encoder_input
        encoder_link = encoder_embedding(encoder_link)
        encoder_link = encoder_gru1(encoder_link)
        encoder_link = encoder_gru2(encoder_link)
        encoder_link = encoder_gru3(encoder_link)
        encoder_output = encoder_link

        # build the decoder, define the placeholders
        decoder_initial_state = Input(shape=(self.state_size,),name='decoder_initial_state')
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(input_dim=self.num_words, output_dim=self.embedding_size, name='decoder_embedding')
        decoder_gru1 = GRU(self.state_size, name='decoder_gru1',
                           return_sequences=True)
        decoder_gru2 = GRU(self.state_size, name='decoder_gru2',
                           return_sequences=True)
        decoder_gru3 = GRU(self.state_size, name='decoder_gru3',
                           return_sequences=True)
        decoder_dense = Dense(self.num_words, activation='linear', name='decoder_output')

        # connect all the placeholders in decoder layer
        decoder_link = decoder_input
        decoder_link = decoder_embedding(decoder_link)
        decoder_link = decoder_gru1(decoder_link, initial_state=decoder_initial_state)
        decoder_link = decoder_gru2(decoder_link, initial_state=decoder_initial_state)
        decoder_link = decoder_gru3(decoder_link, initial_state=decoder_initial_state)
        decoder_output = decoder_dense(decoder_link)

        # connect the model
        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])
        model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])
        model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])

        # finalized the model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        optimizer = RMSprop(lr=self.lr)
        model.compile(optimizer=optimizer, loss="sparse_cross_entropy", target_tensors=[decoder_target])

        # ser early-stopping and tensorboard and save check point
        checkpoint = ModelCheckpoint(filepath=self.save_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
        callbacks = [early_stopping, checkpoint, tensorboard]

        # load check point if exists
        try:
            model.load_weights(self.save_path)
        except Exception as error:
            print("Failed CheckPoint")
            print(error)

        # pack inputs and outputs
        input = {'encoder_input': encoder_input, 'decoder_input': decoder_input}
        output = {'decoder_output': decoder_output}

        # start training
        model.fit(x=input, y=output, batch_size=512, epochs=5, validation_split=self.validation_split, callbacks=callbacks)








