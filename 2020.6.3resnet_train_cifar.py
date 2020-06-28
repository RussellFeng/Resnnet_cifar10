import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import  optimizers, datasets, metrics
import  matplotlib.pyplot as plt
from    resnet_18_34fz import resnet18, resnet34
from    plainnet_18_34fz import plainnet18, plainnet34
from    plainnet50_fz import plainnet50, plainnet152, plainnet101
from    resnet50_fz import resnet50
import  datetime

tf.random.set_seed(2345)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


def preprocess(x, y):
    # [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x,y

def correctnum(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    # [10, b]
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        res.append(correct_k)

    return res

(x,y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)


print(x.shape, y.shape, x_test.shape, y_test.shape)


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(10000).map(preprocess).batch(64).repeat(10)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():

    # [b, 32, 32, 3] => [b, 1, 1, 512]

    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    optimizer = optimizers.Adam(lr=1e-3)
    acc_meter = metrics.Accuracy()
    loss_meter = metrics.Mean()

    for step, (x,y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 512] => [b, 100]
            logits = model(x)
            # [b] => [b, 100]
            y_onehot = tf.one_hot(y, depth=10)
            # compute loss
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            loss_meter.update_state(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if step %10 == 0:
            # model.save_weights('./checkpoints/resnet18')
            # model.save_weights('./checkpoints/resnet34')
            model.save_weights('./checkpoints/resnet50')
            print(step, 'loss:', loss_meter.result().numpy())
            loss_meter.reset_states()
            with summary_writer.as_default():
                tf.summary.scalar('train-loss', float(loss), step=step)


        if step % 50 == 0:
            total_num = 0
            total_correct_top1 = 0
            total_correct_top5 = 0
            acc_meter.reset_states()

            for x,y in test_db:

                logits = model(x)

                prob = tf.nn.softmax(logits, axis=1)
                # pred = tf.argmax(prob, axis=1)
                # pred = tf.cast(pred, dtype=tf.int32)
                # correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
                # correct = tf.reduce_sum(correct)
                correct = correctnum(prob, y, topk=(1,5))

                total_num += x.shape[0]
                total_correct_top1 += int(correct[0])
                total_correct_top5 += int(correct[1])

                # acc_meter.update_state(y, pred)

            acc_top1 = total_correct_top1 / total_num
            acc_top5 = total_correct_top5 / total_num
            print(step, 'acc_top1:', acc_top1)
            print(step, 'acc_top5:', acc_top5)

            with summary_writer.as_default():
                tf.summary.scalar('test-acc_top1', float(acc_top1), step=step)
                tf.summary.scalar('test-acc_top5', float(acc_top5), step=step)
# model.save('model_resnet18.h5')
# model.save('model_resnet34.h5')


if __name__ == '__main__':
    main()

