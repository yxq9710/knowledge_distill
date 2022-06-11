"""
    创建蒸馏实例，在里面定义训练和测试的模式
    定义 student 和 teacher 模型
    获取数据
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Distiller(tf.keras.Model):
    """get a distillation instance, it compiled by train_step and tested by test_step"""

    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distill_loss_fn, alpha, temperature):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_prediction = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_prediction = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_prediction)
            distill_loss = self.distill_loss_fn(tf.nn.softmax(teacher_prediction / self.temperature),
                                                tf.nn.softmax(student_prediction / self.temperature)
                                                )
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_prediction)

        result = {m.name: m.result() for m in self.metrics}
        result.update(
            {"student_loss": student_loss, "distill_loss": distill_loss}
        )
        return result

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def student_model():
    """ get student model """
    student = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ],
        name='student'
    )
    return student


def teacher_model():
    """ get teacher model """
    teacher = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ],
        name='teacher'
    )
    return teacher


def scratch_student_model():
    """ get the clone of student model"""
    return keras.models.clone_model(student_model())


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train.astype("float32") / 255.0, (-1, 28, 28, 1))
    x_test = np.reshape(x_test.astype("float32") / 255.0, (-1, 28, 28, 1))
    return x_train, y_train, x_test, y_test
