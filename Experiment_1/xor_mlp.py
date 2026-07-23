import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Create XOR Dataset
# -------------------------------

np.random.seed(42)

X = np.random.randint(0, 2, (500, 2))

y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32)

# -------------------------------
# Plot XOR Dataset
# -------------------------------

plt.figure(figsize=(5,5))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1],
                    color='red',
                    alpha=0.4,
                    s=40)
    else:
        plt.scatter(X[i,0], X[i,1],
                    color='blue',
                    alpha=0.4,
                    s=40)

plt.title("XOR Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

# -------------------------------
# Build MLP Model
# -------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train
# -------------------------------

history = model.fit(
    X,
    y,
    epochs=1000,
    batch_size=32,
    verbose=1
)

# -------------------------------
# Plot Training Loss
# -------------------------------

plt.figure(figsize=(6,4))

plt.plot(history.history['loss'])

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()

# -------------------------------
# Decision Boundary
# -------------------------------

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
    c=y,
    s=25
)

plt.title("Decision Boundary")

plt.xlabel("X1")

plt.ylabel("X2")

plt.grid(True)

plt.show()

# -------------------------------
# Final Accuracy
# -------------------------------

loss, accuracy = model.evaluate(X, y, verbose=0)

print("\n========================================")
print("Final Accuracy : {:.2f}%".format(accuracy*100))
print("========================================")