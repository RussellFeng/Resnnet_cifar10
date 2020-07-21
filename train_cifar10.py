import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import  optimizers, datasets
from    keras.preprocessing.image import ImageDataGenerator
from    keras.utils import np_utils
import  matplotlib.pyplot as plt
import  datetime
from    resnet_cifar10 import resnet20, resnet32, resnet44, resnet56, resnet110,resnet218
from    plainnet_cifar10 import plainnet20,plainnet32,plainnet44,plainnet56,plainnet110



tf.random.set_seed(2345)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

def acc_top5(y_true, y_pred):

    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

#超参数设定
batch_size = 128
nb_epoch = 100
learning_rate = 0.001
momentum = 0.9
data_augmentation = True

#输入数据参数
nb_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3




(X_train,y_train), (X_test, y_test) = datasets.cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(tf.reduce_min(X_train), tf.reduce_max(X_test))
print(tf.reduce_min(y_train), tf.reduce_max(y_test))
print(np.mean(X_train, axis=0).shape)
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 255.
X_test /= 255.
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



def main():

    # model = tf.keras.models.load_model('model_resnet20.h5')
    # model = tf.keras.models.load_model('model_resnet44.h5')

    #学习率衰减
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=5,
        factor=0.1,
        min_delta=0.001,
        min_lr=0.00001)
    #保存模型参数
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
            histogram_freq=10,
            write_graph=False,
        ),
        learning_rate_reduction,
        model_checkpoint
    ]

    model = resnet20()

    # model.load_weights('./checkpoints/resnet')
    # model.load_weights('./checkpoints/resnet20')
    # model.load_weights('./checkpoints/resnet44')
    # print("weights loaded")
    # model.evaluate(X_test, Y_test, batch_size=batch_size)

    model.build(input_shape=(None, img_rows, img_cols, img_channels))
    model.summary()
    # 可训练参数
    for x in model.trainable_weights:
        print(x.name)
    print('\n')

    # 不可训练参数
    for x in model.non_trainable_weights:
        print(x.name)
    print('\n')
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='resnet34_model.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=True,
    #     dpi=900)
    model.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', acc_top5])

    if not data_augmentation:
        print('Not using data augmentation.')
        model_fit_history = model.fit(X_train, Y_train, batch_size=batch_size,
                                      epochs=nb_epoch,
                                      validation_data=(X_test, Y_test),
                                  validation_freq=1, callbacks=callbacks)

    else:
        print('Using data augmentation.')
        #数据增强方法
        datagen = ImageDataGenerator(
            rotation_range=0,  # 在相同范围内随机旋转图像(degrees, 0 to 180)
            width_shift_range=0.1,  # 随机水平移动图像的宽的比例
            height_shift_range=0.1,  # 随机水垂直移动图像的宽的比例
            horizontal_flip=True,  # 随机水平翻转图片
            vertical_flip=False)  # 随机垂直翻转图片
        datagen.fit(X_train)
        model_fit_history = model.fit(datagen.flow(X_train, Y_train,
                                                   batch_size=batch_size),
                                      steps_per_epoch=
                                      X_train.shape[0] // batch_size,
                                      verbose=1,
                                      epochs=nb_epoch,
                                      max_queue_size=100,
                                      validation_data=(X_test, Y_test),
                                      validation_freq=1, callbacks=callbacks)

    model.evaluate(X_test, Y_test, batch_size=batch_size)

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



# model.save('model_resnet20.h5')
# model.save('model_resnet44.h5')


if __name__ == '__main__':
    main()

