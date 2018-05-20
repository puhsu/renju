import keras
import numpy
import util
from keras.layers import Input, Conv2D, Reshape
from keras.layers import Activation, BatchNormalization
from keras.models import Model




class PolicyNetwork:
    def __init__(self, args):
        self.args = args

        if self.args.modelfile:
            self.model = keras.models.load_model(self.args.modelfile)
        else:
            self.build_network()

        self.callbacks = []

        if self.args.logdir:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=self.args.logdir, batch_size=10))

        if self.args.checkpoints:
            filepath = self.args.checkpoints + '{epoch:02d}.hdf5'
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath))
    

    def build_network(self):
        keras.backend.clear_session()
        self.input_boards = Input(shape=(15, 15, 4))
        x = Conv2D(16, (3, 3), padding='same')(self.input_boards)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))
        
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


    def train(self, train_generator, validation=None, nb_epochs=1, initial_epoch=0):
        '''
        Train for nb_epochs
        '''
        self.model.fit_generator(train_generator.generate(),
                                 steps_per_epoch=train_generator.steps_per_epoch,
                                 epochs=nb_epochs,
                                 initial_epoch=initial_epoch,
                                 callbacks=self.callbacks,
                                 validation_data=validation,
                                 workers=3,
                                 verbose=1)


class RolloutNet:
    def __init__(self, args):
        self.args = args

        if self.args.modelfile:
            self.model = keras.models.load_model(self.args.modelfile)
        else:
            self.build_network()

        self.callbacks = []

        if self.args.logdir:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=self.args.logdir, batch_size=10))

        if self.args.checkpoints:
            filepath = self.args.checkpoints + '{epoch:02d}.hdf5'
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath, period=1))


    def build_network(self):
        keras.backend.clear_session()
        self.input_boards = Input(shape=(15, 15, 4))
        x = Conv2D(16, (3, 3), padding='same')(self.input_boards)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(Activation('relu')(x))        
        x = Conv2D(1, (1, 1), padding='same')(x)

        self.predictions = Activation('softmax')(Reshape((225,))(x))
        self.model = Model(inputs=self.input_boards,
                           outputs=self.predictions)



        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def train(self, train_generator, validation=None, nb_epochs=1, initial_epoch=0):
        '''
        Train for nb_epochs
        '''
        self.model.fit_generator(train_generator.generate(),
                                 steps_per_epoch=train_generator.steps_per_epoch,
                                 epochs=nb_epochs,
                                 initial_epoch=initial_epoch,
                                 callbacks=self.callbacks,
                                 validation_data=validation,
                                 workers=3,
                                 verbose=1)




if __name__ == '__main__':
    args=util.dotdict({
        'modelfile': 'models/model.policy.01.hdf5',
        'logdir': 'tensorboard_logs/',
        'checkpoints': 'models/model.policy.',
        'valid_size': 10000
    })

    games = []
    with open('data/train.renju') as file:
        for game in file:
            res = game.split()[0]
            if res == 'white' or res == 'black':
                games.append(game)

    states_count = 0
    for game in games:
        res, *moves = game.split()
        if res == 'black':
            states_count += int(numpy.ceil(len(moves) / 2))
        if res == 'white':
            states_count += int(numpy.floor(len(moves) / 2))

    print('DONE reading training data. Total states: {states_count}')


    x_validation = []
    y_validation = []
    with open('data/test.renju') as file:
        for game in file.readlines()[:args.valid_size]:
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
                    x_validation.append(numpy.copy(state))
                    y_validation.append(i * 15 + j)
                    state[i, j, 0] = 1
                        
                if player == -1:
                    state[i, j, 1] = 1

                player = -player

    x_validation = numpy.array(x_validation)
    y_validation = numpy.array(y_validation)
    val_len = len(y_validation)
    print(f'DONE preprocessing validation data. States count = {val_len}')

    # train model
    policy = PolicyNetwork(args)

    #train_generator = DataGenerator(games, states_count, batch_size=300, augmentations=True)
    #policy.train(train_generator,
    #             validation=(x_validation, y_validation),
    #             nb_epochs=1)

    train_generator = DataGenerator(games, states_count, batch_size=600, augmentations=True)
    policy.train(train_generator,
                 validation=(x_validation, y_validation),
                 nb_epochs=1)

    train_generator = DataGenerator(games, states_count, batch_size=1200, augmentations=True)
    policy.train(train_generator,
                 validation=(x_validation, y_validation),
                 nb_epochs=1)

