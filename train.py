from absl import flags
from absl import app
import tensorflow as tf
from runner import train_eval_runner

flags.DEFINE_integer('Epoch_num', 200,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('base_dir', 'save_data',
                    'Base directory to host all required sub-directories.')
FLAGS = flags.FLAGS

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    runner = train_eval_runner(tf.Session(), FLAGS.Epoch_num, FLAGS.base_dir)
    runner.run_train_eval()


if __name__ == '__main__':
    app.run(main)