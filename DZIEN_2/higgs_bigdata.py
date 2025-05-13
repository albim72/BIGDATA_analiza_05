#Sprawdzenie dostępności GPU
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#Pobranie danych HIGGS (~2.5 GB)
!wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
!gunzip -f HIGGS.csv.gz

#Pipeline danych
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

train_ds = dataset.take(1000)  # ≈ 1 mln przykładów


#Model na GPU
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURE_DIM,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Trening (na GPU)
model.fit(train_ds, epochs=5)

#Ewaluacja (opcjonalnie)
eval_ds = dataset.skip(1000).take(200)
model.evaluate(eval_ds)

