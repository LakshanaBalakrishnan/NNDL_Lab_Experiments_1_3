import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# AND Dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=np.float32)

y = np.array([
    [0],
    [0],
    [0],
    [1]
], dtype=np.float32)

# Build Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X,
    y,
    epochs=1000,
    verbose=0
)

# Dataset Plot
plt.figure(figsize=(5,5))

for i in range(len(X)):
    color = "blue" if y[i]==1 else "red"
    plt.scatter(X[i,0], X[i,1], c=color, s=120)

plt.title("AND Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

# Training Loss
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Decision Boundary
xx, yy = np.meshgrid(
    np.linspace(-0.2,1.2,200),
    np.linspace(-0.2,1.2,200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
pred = model.predict(grid, verbose=0)
pred = pred.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, pred, levels=[0,0.5,1], alpha=0.4)

plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=120)

plt.title("Decision Boundary (AND)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

# Predictions
predictions = (model.predict(X) > 0.5).astype(int)

print("\nAND Predictions")
print("----------------")

for i in range(4):
    print(
        X[i],
        "Expected:", y[i][0],
        "Predicted:", predictions[i][0]
    )

loss, acc = model.evaluate(X, y, verbose=0)

print("\nAccuracy = {:.2f}%".format(acc*100))