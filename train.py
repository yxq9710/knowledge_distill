from utils import *
from config import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy

teacher = teacher_model()
student = student_model()
student_scratch = scratch_student_model()

x_train, y_train, x_test, y_test = get_data()

teacher.compile(optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy()],
                )
teacher.fit(x_train, y_train, batch_size=batch_size, epochs=teacher_epochs)
teacher_loss, teacher_acc = teacher.evaluate(x_test, y_test)
print("teacher_acc : ", teacher_acc)

distiller = Distiller(student, teacher)
distiller.compile(optimizer=Adam(),
                  metrics=[SparseCategoricalAccuracy()],
                  student_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
                  distill_loss_fn=tf.keras.losses.KLDivergence(),
                  alpha=alpha,
                  temperature=temperature)
distiller.fit(x_train, y_train, batch_size=batch_size, epochs=student_epochs)
distill_acc, distill_loss = distiller.evaluate(x_test, y_test)
print("distill_acc : ", distill_acc)

student_scratch.compile(optimizer=Adam(),
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        metrics=[SparseCategoricalAccuracy()],
                        )
student_scratch.fit(x_train, y_train, batch_size=batch_size, epochs=student_epochs)
scratch_loss, scratch_acc = student_scratch.evaluate(x_test, y_test)
print("scratch_acc : ", scratch_acc)
print("end")
