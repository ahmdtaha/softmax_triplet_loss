import tensorflow as tf

class ConvEmbed(object):
    def name(self):
        return "ConvTSN"

    def __init__(self, emb_dim=256, n_input=1536, n_h=8, n_w=8):

        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.emb_dim = emb_dim



        self.W = tf.get_variable(name="W", shape=[self.n_input * n_h * n_w, self.emb_dim],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(1.),
                                 trainable=True)
        self.b = tf.get_variable(name="b", shape=[self.emb_dim],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

    def forward(self, x):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """
        x_emb = tf.reshape(x, [-1, self.n_h * self.n_w * self.n_input])
        embedding = tf.nn.xw_plus_b(x_emb, self.W, self.b)
        return embedding



