import tensorflow as tf
import os

class PolicyNet:
    '''
    Policy network class. Used for fiting and evaluating model.
    '''
    def __init__(self):
        self.training_summary_writer = None
        self.training_stats = StatisticsCollector()

        self.session = tf.Session()
        self.build_graph()


    def build_graph(self, lr=1e-4):
        Conv = tf.layers.conv2d
        Relu = tf.nn.relu
        BatchNorm = tf.layers.batch_normalization
        
        # variables
        self.input_boards = tf.placeholder(tf.float32, shape=[None, 15, 15, 9], name="input_boards")
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # layers
        h_conv1 = Conv(self.input_boards, filters=16, kernel_size=[3,3], padding='same')
        h_conv2 = Relu(BatchNorm(Conv(h_conv1, filters=16, kernel_size=[3,3], padding='same'), training=self.is_training))
        h_conv3 = Conv(h_conv2, filters=32, kernel_size=[3,3], padding='same')
        h_conv4 = Relu(BatchNorm(Conv(h_conv3, filters=32, kernel_size=[3,3], padding='same'), training=self.is_training))
        h_conv5 = Conv(h_conv4, filters=64, kernel_size=[3,3], padding='same')
        h_conv6 = Relu(BatchNorm(Conv(h_conv5, filters=64, kernel_size=[3,3], padding='same'), training=self.is_training))
        h_conv7 = Conv(h_conv6, filters=1, kernel_size=[1,1], padding='same')
        h_conv7_flat = tf.reshape(h_conv7, [-1, 15 * 15])

        # predicted probabilities
        self.logits = tf.nn.softmax(h_conv7_flat)

        # loss & train step & accuracy
        self.log_likelihood_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.log_likelihood_cost)
        
        correct_pred = tf.equal(tf.argmax(self.logits, axis=1, output_type=tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # utilitis
        self.saver = tf.train.Saver()

    
    def get_global_step(self):
        return self.session.run(self.global_step)

    
    def init_logging(self, tensorboard_logdir):
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)
        
        
    def init_variables(self, filepath):
        self.session.run(tf.global_variables_initializer())
        if filepath is not None:
            print(f'Loading model from {filepath}')
            self.saver.restore(self.session, filepath)
    
    
    def save_model(self, filepath):
        if filepath is not None:
            print(f'Saving model to {filepath}')
            self.saver.save(self.session, filepath)
    
    
    def print_summary(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            
            for dim in shape:
                variable_parameters *= dim.value
            
            total_parameters += variable_parameters
        print(f'Network has {total_parameters} trainable parameters.')


    def gen_minibatch(self, X, y, batch_size):
        l = len(X)
        for ndx in range(0, l, batch_size):
            yield X[ndx:min(ndx + batch_size, l)], y[ndx:min(ndx + batch_size, l)]


    def fit(self, X, y, epochs=1, batch_size=300, verbose=False, interval=10000):        
        for i in range(epochs):
            step = 0
            for X_batch, y_batch in self.gen_minibatch(X, y, batch_size):
                _, acc, cost = self.session.run([self.train_op, self.accuracy, self.log_likelihood_cost], 
                                                 feed_dict={self.input_boards: X_batch,
                                                            self.labels: y_batch,
                                                            self.is_training: True})
                self.training_stats.report(acc, cost)
                step += 1
                if verbose and step % interval == 0:
                    print(f'|__ FINISHED STEP {step}')


            avg_acc, avg_cost, accuracy_summaries = self.training_stats.collect()
            global_step = self.get_global_step()
            if verbose:
                print(f'\n✔ EPOCH {i}. Step {global_step} of training. accuracy: {avg_acc} cost: {avg_cost}.')
            
            if self.training_summary_writer is not None:
                self.training_summary_writer.add_summary(accuracy_summaries, global_step)


    def predict(self, X):
        labels = self.session.run(self.logits,
                                  feed_dict={self.input_boards: X,
                                             self.is_training: False})
        return labels


    def accuracy_score(self, X, y):
        acc = self.session.run([self.accuracy],
                               feed_dict={self.input_boards: X,
                                          self.labels: y,
                                          self.is_training: False})
        return acc


    def reset(self):
        self.session.close()
        self.__init__()


class StatisticsCollector:
    '''
    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then shove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    '''
    graph = tf.Graph()
    with tf.device('/cpu:0'), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary


if __name__ == "__main__":
    test_net = PolicyNet()
    test_net.init_logging("./tensorboard_logs")
    test_net.print_summary()
    
