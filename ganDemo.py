import os
import time
import random

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers

BUFFER_SIZE = 60000
BATCH_SIZE = 256
CHECK_POINT_DIR = './training'
CHECK_POINT_PREFIX = os.path.join(CHECK_POINT_DIR, 'checkpoints')
EPOCH = 60
NOISE_DIM = 100
NUM_GENERATE_EXAMPLE = 16
SHUFFLE_SEED = random.randint(0, 65535)

# 将所有GPU显存的alloc模式换成随用随alloc
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


(trainX, trainY), (_, _) = tf.keras.datasets.mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')
trainX = (trainX - 127.5) / 127.5
trainY = tf.one_hot(trainY, 10)
trainDataS = tf.data.Dataset.from_tensor_slices((trainX, trainY)).shuffle(BUFFER_SIZE, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE)


def make_generator():
    return tf.keras.Sequential([
        # (_,100,)
        layers.Dense(7 * 7 * 256, use_bias=False,
                     input_shape=(NOISE_DIM,)),
        # (_,7*7*256)
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        # (_,7,7,256)
        layers.Conv2DTranspose(
            128, (5, 5), padding='same', use_bias=False),
        # (_,7,7,128)
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(
            2, 2), padding='same', use_bias=False),
        # (_,14,14,64)
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False)
        # (_,28,28,1)
    ])


def make_discriminator():
    return tf.keras.Sequential([
        # (_,28,28,1)
        layers.Conv2D(64, (5, 5), strides=(2, 2),
                      padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(.3),
        # (_,14,14,64)
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(.3),
        # (_,7,7,128)
        layers.Flatten(),
        # (_,7*7*128)
        layers.Dense(10),
        # (_,10)
    ])


crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def discriminator_loss(real_output, real_label, fake_output):
    real_loss = crossEntropy(real_label, real_output)
    fake_loss = crossEntropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


@tf.function
def generator_loss(fake_output, fake_label):
    return crossEntropy(fake_label, fake_output)


if __name__ == '__main__':
    gen = make_generator()
    dis = make_discriminator()

    genOptim = tf.keras.optimizers.Adam(1e-4)
    disOptim = tf.keras.optimizers.Adam(1e-4)
    checkPoint = tf.train.Checkpoint(
        gen=gen, dis=dis, genOptim=genOptim, disOptim=disOptim)

    @tf.function
    def train_step(images, labels):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM - 10])
        ctrl = tf.one_hot(tf.random.uniform(
            [BATCH_SIZE], minval=0, maxval=10, dtype=tf.int32), 10)
        noise = tf.concat([ctrl, noise], 1)

        with tf.GradientTape() as genTape, tf.GradientTape() as disTape:
            genImgs = gen(noise, training=True)

            fakeOutput = dis(genImgs, training=True)
            realOutput = dis(images, training=True)

            genLoss = generator_loss(fakeOutput, ctrl)
            disLoss = discriminator_loss(realOutput, labels, fakeOutput)

        genGrad = genTape.gradient(genLoss, gen.trainable_variables)
        disGrad = disTape.gradient(disLoss, dis.trainable_variables)

        genOptim.apply_gradients(zip(genGrad, gen.trainable_variables))
        disOptim.apply_gradients(zip(disGrad, dis.trainable_variables))

    def generate_and_save_images(model, epoch, test_input):
        # 注意 training` 设定为 False
        # 因此，所有层都在推理模式下运行（batchnorm）。
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] *
                       127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('imgs/image_at_epoch_{:04d}.png'.format(epoch))
        # plt.show()
        plt.close(fig)

    def train(dataset, epochs):
        # 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
        seed = tf.random.normal([NUM_GENERATE_EXAMPLE, NOISE_DIM - 10])  # 随机部分
        ctrl = tf.concat([tf.one_hot(range(10), 10), tf.random.normal(
            [NUM_GENERATE_EXAMPLE - 10, 10])], 0)
        seed = tf.concat([ctrl, seed], 1)

        for epoch in range(epochs):
            start = time.time()

            for image_batch, label_batch in dataset:
                train_step(image_batch, label_batch)

            generate_and_save_images(gen,
                                     epoch + 1,
                                     seed)

            # 每 15 个 epoch 保存一次模型
            if (epoch + 1) % 15 == 0:
                checkPoint.save(file_prefix=CHECK_POINT_PREFIX)

            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time() - start))

        generate_and_save_images(gen,
                                 epochs,
                                 seed)

    checkPoint.restore(tf.train.latest_checkpoint(CHECK_POINT_DIR))
    train(trainDataS, EPOCH)
