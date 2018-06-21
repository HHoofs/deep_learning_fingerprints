import tensorflow as tf
from tf_utils import Dense, CNN

class NeuralNet_Classification:

    def __init__(self, imsize, batchgen):

        self.imsize = imsize

        self.batchgen = batchgen

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        # Feed placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, imsize, imsize, 1], name='input')
        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        # Standardization
        self.x_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x)

        # Run the network
        self.cnn_output = CNN(self.x_standardized, self.dropout_rate)

        self.fc1 = Dense(self.cnn_output, 256, tf.nn.relu)
        self.fc1 = tf.layers.dropout(inputs=self.fc1, rate=self.dropout_rate)

        self.fc2 = Dense(self.fc1, 256, tf.nn.relu)
        self.logits = Dense(self.fc2, 5, None)
        self.prediction = tf.nn.softmax(self.logits)

        self.label = tf.placeholder(tf.int32, [None, 5])
        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.label)


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    name='checkpoint_saver')


    def train(self, num_steps, batch_size, dropout_rate, lr, decay, augment, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []

        for step in range(num_steps):


            x_batch, y_batch = self.batchgen.generate_train_batch(batch_size, augment)
            feed_dict = {
                        self.x: x_batch,
                        self.label: y_batch,
                        self.dropout_rate: dropout_rate,
                        self.lr: lr
                        }

            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 100 == 0:
                x_batch, y_batch = self.batchgen.generate_val_batch(batch_size, False)
                feed_dict = {
                            self.x: x_batch,
                            self.label: y_batch,
                            self.dropout_rate: 0
                            }

                val_loss = self.session.run([self.loss], feed_dict=feed_dict)
                val_loss_list.append(val_loss)
                loss_list.append(loss_)
                print('step: {}'.format(step))
                print('train loss: {}'.format(loss_))
                print('val loss: {}'.format(val_loss))
                print('lr: {}'.format(lr))
                print('')

            if (step + 1) % 1000 == 0 or step == num_steps - 1:
                self.saver.save(self.session, checkpoint + str(step) + '.ckpt')
                print('Saved to {}'.format(checkpoint + str(step) + '.ckpt'))

        return loss_list, val_loss_list


    def predict(self, image):

        feed_dict = {
            self.x: image.reshape(1, self.imsize, self.imsize, 1),
            self.dropout_rate: 0
        }
        pred = self.session.run([self.prediction], feed_dict=feed_dict)

        return pred[0][0]


    def load_weights(self, path):
        self.saver.restore(self.session, path)
        print('Weights loaded.')