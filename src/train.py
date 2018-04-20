import util
import nnet
import argparse
import numpy as np

CHUNKS_COUNT = 20

def preprocess_games(games):
    '''
    filter draws and store [state, action] pairs for every step in every game
    also add features:
                                                                    Channel indices:
        - your turns                                                 0
        - opponent turns                                             1
        - 4 channels history for your turns                          2, 4, 6
        - 4 channels history for opponent turns                      3, 5, 7
        - color channel (all ones if playing for black)              8

    
    returns:
        states -- nd array of shape [games, 15, 15, 9]
        labels -- nd array of [games, 1]
    '''
    states = []
    labels = []
    
    count = 0
    
    for game in games:
        res, *moves = game.split()
        state = np.zeros((15, 15, 9), dtype=np.uint8)

        player_prev_move = None
        opponent_prev_move = None
        
        if res == 'white':
            player = -1
        elif res == 'black':
            state[..., 8] = 1
            player = 1
        else:
            continue

        for move in moves:
            count += 1
            i, j = util.to_pos(move)

            if player == 1:
                # make move
                states.append(np.copy(state))
                labels.append(i * 15 + j)

                # move history
                np.copyto(state[..., 6], state[..., 4])
                np.copyto(state[..., 4], state[..., 2])

                if player_prev_move:
                    m, n = util.to_pos(player_prev_move)
                    state[..., 2] = 0
                    state[m, n, 2] = 1

                state[i, j, 0] = 1
                player_prev_move = move

            if player == -1:
                np.copyto(state[..., 7], state[..., 5])
                np.copyto(state[..., 5], state[..., 3])

                if opponent_prev_move:
                    m, n = util.to_pos(opponent_prev_move)
                    state[..., 3] = 0
                    state[m, n, 3] = 1

                state[i, j, 1] = 1
                opponent_prev_move = move

            player = -player

    print(f'Finished preprocessing. Loaded {count} game moves')
    return np.array(states), np.array(labels, dtype=np.uint8)


def data_iter(num_chunks):
    '''
    Iterate over num_chunks files yield results of preprocessing.
    '''
    for i in range(num_chunks):
        with open(f'./data/train-chunk-{i:02d}') as games:
            yield preprocess_games(games)


def train_nnet(epochs, num_chunks, save_path, load_path, logdir, verbose):
    policynet = nnet.PolicyNet()
    policynet.init_variables(load_path)

    if logdir:
        policynet.init_logging(logdir)

    for X_train, y_train in data_iter(num_chunks):
        policynet.fit(X_train, y_train, epochs=epochs, verbose=verbose, interval=10)

    if save_path:
        policynet.save_model(save_path)

    return policynet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Policy Network')
    parser.add_argument('epochs', type=int, help='Number of epochs to train for', default=1)
    parser.add_argument('chunks', type=int, help='Number of files to train on', default=20)
    parser.add_argument('--save', type=str, help='Filename for model snapshot',)
    parser.add_argument('--load', type=str, help='File with trained model')
    parser.add_argument('--logs', type=str, help='Logging directory', default='data/tensorboard_logs')
    parser.add_argument('--verbose', type=int, choices=(0,1), help='True to print info', default=True)

    args = parser.parse_args()

    train_nnet(epochs=args.epochs,
               num_chunks=args.chunks,
               save_path=args.save,
               load_path=args.load,
               logdir=args.logs,
               verbose=args.verbose)

    
    
