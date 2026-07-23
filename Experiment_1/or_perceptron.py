import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# OR Dataset
# ---------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [1]
], dtype=np.float32)

# ---------------------------------
# Build Single-Layer Perceptron
# ---------------------------------

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

# ---------------------------------
# Plot Dataset
# ---------------------------------

plt.figure(figsize=(5,5))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1],
                    color='red',
                    s=120,
                    label='Class 0')
    else:
        plt.scatter(X[i,0], X[i,1],
                    color='blue',
                    s=120,
                    label='Class 1' if i==1 else "")

plt.title("OR Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------------
# Plot Training Loss
# ---------------------------------

plt.figure(figsize=(6,4))

plt.plot(history.history["loss"])

plt.title("Training Loss (OR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.show()

# ---------------------------------
# Decision Boundary
# ---------------------------------

xx, yy = np.meshgrid(
    np.linspace(-0.2,1.2,200),
    np.linspace(-0.2,1.2,200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

pred = model.predict(grid, verbose=0)
pred = pred.reshape(xx.shape)

plt.figure(figsize=(6,6))

plt.contourf(
    xx,
    yy,
    pred,
    levels=[0,0.5,1],
    alpha=0.4
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y.flatten(),
    s=120
)

plt.title("Decision Boundary (OR)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)

plt.show()

# ---------------------------------
# Predictions
# ---------------------------------

predictions = (model.predict(X) > 0.5).astype(int)

print("\n==============================")
print("OR Gate Predictions")
print("==============================")

for i in range(len(X)):
    print(
        f"Input : {X[i]}",
        f" Expected : {int(y[i][0])}",
        f" Predicted : {predictions[i][0]}"
    )

loss, accuracy = model.evaluate(X, y, verbose=0)

print("\n==============================")
print(f"Final Accuracy : {accuracy*100:.2f}%")
print("==============================")