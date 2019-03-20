from train_data import predict_train_data
from predict_agent import predict_agent
import tensorflow as tf
import numpy as np
import os
slim = tf.contrib.slim

class train_eval_runner():

    def __init__(self,
                 sess,
                 epochs,
                 base_dir,
                 codes=['NQH']): # codes=['ESH', 'HSIH', 'NQH']
        self.checkpoint_path = os.path.join(base_dir, './checkpoints/predict.ckpt')
        self.base_dir = base_dir
        self.logdir = base_dir
        self.sess = sess
        self.codes = codes
        self.epochs = epochs
        self.predict_train_datas = [predict_train_data(context, 'train') for context in self.codes]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        self.agent = predict_agent(self.sess,
                                   self.predict_train_datas[0].get_input_shape(),
                                   self.predict_train_datas[0].get_label_shape(),
                                   self.predict_train_datas[0].get_batch_size(),
                                   False if ckpt == None else True)


        self.eval_loss_ph = tf.placeholder(tf.float32, [None], name='eval_loss_ph')
        self.train_loss_ph = tf.placeholder(tf.float32, [None], name='train_loss_ph')
        tf.summary.scalar('Train_loss', tf.reduce_mean(self.train_loss_ph))
        tf.summary.scalar('eval_loss', tf.reduce_mean(self.eval_loss_ph))

        self.writer = tf.summary.FileWriter(self.logdir, sess.graph)
        self.summary_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        if ckpt is None:
            sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def run_train_eval(self):
        for epoch in range(self.epochs):
            train_loss_list = []
            eval_loss_list = []
            for data in self.predict_train_datas:
                data.set_mode('train')
                while(True):
                    batch_data, batch_label = data.get_input_batch_data()
                    train_loss_list.append(self.agent.train(batch_data, batch_label))

                    done = data.next_data()
                    if done:
                        break
                data.reset()

            if epoch < 5:
                continue

            predict = 0
            label = 0
            for data in self.predict_train_datas:
                data.set_mode('eval')
                while(True):
                    batch_data, batch_label = data.get_input_batch_data()
                    temp_predict, loss = self.agent.predict(batch_data, batch_label)
                    label = batch_label[0][0] * data.norm_max
                    predict = temp_predict[0][0] * data.norm_max
                    eval_loss_list.append(loss)

                    done = data.next_data()
                    if done:
                        break
            eval_loss_list = np.reshape(eval_loss_list, [-1])
            summary = self.sess.run(self.summary_merged, feed_dict={self.eval_loss_ph: eval_loss_list,
                                                                               self.train_loss_ph: train_loss_list})
            self.writer.add_summary(summary, epoch)

            loss_mean = np.mean(train_loss_list)
            accuracy_mean = np.mean(eval_loss_list)
            tf.logging.info(f"epoch : {epoch}, train loss : {str(loss_mean)}, eval loss : {str(accuracy_mean)}, predict = {predict}, label = {label}")

        # self.saver.save(self.sess, self.checkpoint_path)