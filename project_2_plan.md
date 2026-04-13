# 🗺️ ROADMAP FOR `.ipynb` NOTEBOOK PROJECT

## 🟢 Phase 1 — Notebook Structure Design (START HERE)

### 🎯 Goal:

Plan your notebook like a **story** (this is very important for grading).

### Your notebook should have this structure:

```text
1. Title & Introduction
2. Imports & Setup
3. Dataset Loading
4. Data Preprocessing
5. CNN Model Design
6. Model Training
7. Evaluation
8. Visualization
9. Testing Predictions
10. Conclusion
```

👉 Each section = **Markdown cell + Code cells**

---

## 🔵 Phase 2 — Setup & Imports

### 📓 Notebook Cells:

### Markdown:

Explain:

* What the project is
* What tools you use

### Code:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

---

## 🟡 Phase 3 — Dataset Section

Use:
👉 MNIST

### Markdown:

Explain:

* Number of images
* Size (28×28)
* Labels (0–9)

### Code:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

## 🟣 Phase 4 — Data Preprocessing

### Markdown:

Explain:

* Normalization
* Reshaping

### Code:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

---

## 🔴 Phase 5 — CNN Model (Core Section)

### Markdown (VERY IMPORTANT ⚠️):

Explain each:

* Conv2D
* ReLU
* MaxPooling
* Flatten
* Dense

### Code:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## 🟠 Phase 6 — Compile & Train

### Markdown:

Explain:

* Loss function
* Optimizer
* Epochs

### Code:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

---

## 🟤 Phase 7 — Evaluation

### Markdown:

Explain:

* What accuracy means

### Code:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
```

---

## ⚫ Phase 8 — Visualization (VERY IMPORTANT)

### Markdown:

Explain:

* Why graphs matter

### Code:

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.show()
```

---

## ⚪ Phase 9 — Predictions

### Markdown:

Explain:

* Show how model predicts

### Code:

```python
import numpy as np

pred = model.predict(x_test[:5])
print(np.argmax(pred, axis=1))
```

---

## 🧾 Phase 10 — Conclusion

### Markdown only:

Write:

* What you achieved
* Accuracy
* Possible improvements
