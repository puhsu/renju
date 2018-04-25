import keras
import numpy
import util
from keras.layers import Input, Conv2D, Reshape
from keras.layers import Activation, BatchNormalization
from keras.models import Model



class DataGenerator:
    '''
    Generates sequence of state action pairs from games log file
    '''
    def __init__(self, games, total_examples, batch_size=32):
        self.games = games
        self.total_examples = total_examples
        self.batch_size = batch_size
        self.steps_per_epoch = int(numpy.floor(self.total_examples / self.batch_size))


    def generate(self):
        'Generate one batch of data'
        X_batch = []
        y_batch = []
        count = 0
        while True:
            for game in self.games:
                res, *moves = game.split()
                state = numpy.zeros((15, 15, 4), dtype=numpy.uint8)

                if res == 'white':
                    player = -1
                elif res == 'black':
                    state[..., 2] = 1
                    player = 1
                else:
                    continue

                for move in moves:
                    i, j = util.to_pos(move)
                    
                    if player == 1:
                        X_batch.append(numpy.copy(state))
                        y_batch.append(i * 15 + j)
                        count += 1
                        if count == self.batch_size:
                            yield numpy.array(X_batch), numpy.array(y_batch)
                            X_batch = []
                            y_batch = []
                            count = 0
                        state[i, j, 0] = 1
                        
                    if player == -1:
                        state[i, j, 1] = 1

                    player = -player





class PolicyNetwork:
    def __init__(self, args):
        self.args = args

        if self.args.modelfile:
            self.model = keras.models.load_model(modelfile)
        else:
            self.build_network()

        self.callbacks = []
        # logging
        if self.args.logdir:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=self.args.logdir))
        # model saving
        if self.args.checkpoints:
            filepath = self.args.checkpoints + '{epoch:02d}.hdf5'
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath))
    

    def build_network(self):
        # clear previous graphs and close session
        keras.backend.clear_session()


        self.input_boards = Input(shape=(15, 15, 4))
        x = Conv2D(16, (3, 3), padding='same')(self.input_boards)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))
        x = Conv2D(1, (1, 1), padding='same')(x)

        self.predictions = Activation('softmax')(Reshape((225,))(x))
        self.model = Model(inputs=self.input_boards,
                           outputs=self.predictions)



        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def train(self, train_generator, valid_generator=None, nb_epochs=1):
        '''
        Train for nb_epochs
        '''

        
        self.model.fit_generator(train_generator.generate(),
                                 steps_per_epoch=train_generator.steps_per_epoch,
                                 epochs=nb_epochs,
                                 callbacks=self.callbacks,
                                 validation_data=valid_generator,
                                 workers=2,
                                 verbose=1)
    
        
if __name__ == '__main__':
    args=util.dotdict({
        'modelfile': None,
        'logdir': 'tensorboard_logs/',
        'checkpoints': 'model',
    })

    games = []
    with open('data/raw/train.renju') as file:
        for game in file:
            res, *_ = game.split()
            if res == 'white' or res == 'black':
                games.append(game)

    states_count = 0
    for game in games:
        res, *moves = game.split()
        if res == 'black':
            states_count += int(numpy.ceil(len(moves) / 2))
        if res == 'white':
            states_count += int(numpy.floor(len(moves) / 2))

    train_generator = DataGenerator(games, states_count, batch_size=300)
    policy = PolicyNetwork(args)
    policy.train(train_generator, nb_epochs=3)

