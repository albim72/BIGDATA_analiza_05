
import tensorflow as tf
import os

# Sprawdzenie GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Pobranie danych
import urllib.request
import gzip

if not os.path.exists("HIGGS.csv"):
    print("Pobieranie danych...")
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz", "HIGGS.csv.gz"
    )
    with gzip.open("HIGGS.csv.gz", "rb") as f_in, open("HIGGS.csv", "wb") as f_out:
        f_out.write(f_in.read())

# Pipeline danych
CSV_FILE = "HIGGS.csv"
FEATURE_DIM = 28
BATCH_SIZE = 1024

def parse_csv(line):
    defaults = [[0.0]] * (FEATURE_DIM + 1)
    fields = tf.io.decode_csv(line, record_defaults=defaults)
    label = fields[0]
    features = tf.stack(fields[1:])
    return features, label

dataset = (
    tf.data.TextLineDataset(CSV_FILE)
    .map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(100_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = dataset.take(1000)           # ~1M próbek
val_ds   = dataset.skip(1000).take(200) # ~0.2M próbek
test_ds  = dataset.skip(1200).take(200) # ~0.2M próbek

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURE_DIM,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Callbacki
log_dir = "logs/higgs"
os.makedirs(log_dir, exist_ok=True)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=3,
    restore_best_weights=True,
    mode='max'
)

# Trening
model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[tensorboard_cb, earlystop_cb]
)

# Ewaluacja
loss, acc, auc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
