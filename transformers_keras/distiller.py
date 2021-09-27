import tensorflow as tf


class Distiller(tf.keras.Model):
    """Model distiller."""

    def __init__(self, teacher: tf.keras.Model, student: tf.keras.Model, **kwargs):
        super().__init__(inputs=student.inputs, outputs=student.outputs, **kwargs)
        self.teacher = teacher
        self.teacher.trainable = False
        self.student = student

    def compile(self, student_loss, distill_loss, optimizer, metrics=None, alpha=0.1, temperature=3, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.student_loss = student_loss
        self.distill_loss = distill_loss
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            # Compute losses
            _student_loss = self.student_loss(y, student_predictions)
            _distill_loss = self.distill_loss(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * _student_loss + (1 - self.alpha) * _distill_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": _student_loss, "distill_loss": _distill_loss})
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_prediction = self.student(x, training=False)
        # Calculate the loss
        _student_loss = self.student_loss(y, y_prediction)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": _student_loss})
        return results
