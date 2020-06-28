import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses,datasets, metrics
from    tensorflow.keras.callbacks import EarlyStopping
import  datetime
import  matplotlib.pyplot as plt

tf.random.set_seed(2345)
np.random.seed(2345)

assert tf.__version__.startswith('2.')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

def preprocess(x,y):
    # [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)

    return x, y

def acc_top5(y_true, y_pred):

    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


(x,y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)

print(x.shape, y.shape, x_test.shape, y_test.shape)

batchsz = 64
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(10000).map(preprocess).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

def main():

    learning_rate = 1e-3

    #学习率衰减
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        factor=0.1,
        min_delta=0.001,
        min_lr=0.0001)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/resnet',
        verbose=0,
        save_weights_only=True)
    # model_checkpoint = keras.callbacks.ModelCheckpoint(
    #     filepath='/checkpoints/plainnet',
    #     verbose=0,
    #     save_weights_only=True)
    # 并且作为callbacks进入generator,开始训练

    # early_stopping = EarlyStopping(
    #     monitor='val_accuracy',
    #     min_delta=0.001,
    #     patience=5
    # )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=False,
        ),
        learning_rate_reduction,
        model_checkpoint
    ]




    # 迁移学习
    model = keras.applications.ResNet50(weights='imagenet', include_top=False)
    model.trainable = True
    newmodel = keras.Sequential([
        model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10,
                     kernel_regularizer=
                     keras.regularizers.l2(0.0001))
    ])
    newmodel.build(input_shape=(None,32,32,3))
    newmodel.summary()




    newmodel.compile(optimizer=optimizers.Adam(lr=learning_rate),
                   loss=losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy', acc_top5])
    model_fit_history = newmodel.fit(train_db, validation_data=test_db, validation_freq=1,
                 epochs=50, callbacks=callbacks)
    newmodel.evaluate(test_db)

    # 绘制训练 & 验证的loss
    plt.plot(model_fit_history.history['loss'])
    plt.plot(model_fit_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的accuracy
    plt.plot(model_fit_history.history['accuracy'])
    plt.plot(model_fit_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()