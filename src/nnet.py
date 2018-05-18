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
    def __init__(self, games, total_examples, batch_size=300, augmentations=True):
        self.games = games
        self.total_examples = total_examples
        self.batch_size = batch_size
        self.steps_per_epoch = int(numpy.floor(self.total_examples / self.batch_size))
        self.augmentations = augmentations
        if self.augmentations:
            self.get_rotation_map()


    def generate(self):
        '''
        yield one batch of data
        '''
        x_batch = []
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
                        if self.augmentations:
                            aug_state, (y, x) = self.augment_state(numpy.copy(state), (i, j)) 
                            x_batch.append(aug_state)
                            y_batch.append(y * 15 + x)
                        else:
                            x_batch.append(numpy.copy(state))
                            y_batch.append(i * 15 + j)

                        count += 1
                        if count == self.batch_size:
                            yield numpy.array(x_batch), numpy.array(y_batch)
                            x_batch = []
                            y_batch = []
                            count = 0
                        state[i, j, 0] = 1
                        
                    if player == -1:
                        state[i, j, 1] = 1

                    player = -player


    def get_rotation_map(self):
        '''creates array with pos (i, j) -> (i', j') maping after k rotations'''
        self.rotation_map = numpy.zeros((4, 15, 15, 2), dtype=int)
        for m in range(225):
            i, j = m // 15, m % 15
            state = numpy.zeros((15, 15))
            state[i, j] = 1
            for k in range(4):
                rotated_state = numpy.rot90(state, k)
                pos = numpy.argmax(rotated_state)
                self.rotation_map[k][i, j] = pos // 15, pos % 15


    def augment_state(self, state, pos):
        '''
        Randomly shift and/or rotate and/or flip
        '''
        # find borders
        board = state[..., 0] + state[..., 1]
        black = state[0, 0, 2]
        (min_x, max_x), (min_y, max_y) = (0, 14), (0, 14)
        ind_y, ind_x = numpy.nonzero(board)

        if ind_x.size > 0:
            min_x, max_x = numpy.min(ind_x), numpy.max(ind_x)
            min_x = min(min_x, pos[1])
            max_x = max(max_x, pos[1])
        if ind_y.size > 0:
            min_y, max_y = numpy.min(ind_y), numpy.max(ind_y)
            min_y = min(min_y, pos[0])
            max_y = max(max_y, pos[0])
    
        # do shift
        if numpy.random.uniform(0, 1) < .8:
            x_shift = numpy.random.randint(-min_x, 15 - max_x)
            y_shift = numpy.random.randint(-min_y, 15 - max_y)
            state = numpy.roll(state, (y_shift, x_shift), axis=(0, 1))
            pos = pos[0] + y_shift, pos[1] + x_shift
        
        # do rotation
        rotations_n = numpy.random.randint(4)
        state = numpy.rot90(state, rotations_n, axes=(0, 1))
        pos = tuple(self.rotation_map[rotations_n][pos])
    
        # do reflection
        if numpy.random.uniform(0, 1) < .5:
            state = numpy.fliplr(state)
            pos = pos[0], 14 - pos[1]

        if black:
            state[..., 2] = 1

        return state, pos
            


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

