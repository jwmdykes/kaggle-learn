# %%
from pickletools import optimize
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.compose import make_column_transformer, make_column_selector  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
import sklearn  # type: ignore
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
# %% [markdown]
# We now create a network with 1 linear unit
# %%
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
# %%
# view the model
model.summary()
# %%
# setup plotting
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


# %%
red_wine = pd.read_csv("../data/red-wine.csv")
red_wine.head()

# %%
red_wine.shape
# %%
input_shape = (11,)

# %%
input_layer = keras.layers.Input(shape=input_shape)
output_layer = keras.layers.Dense(1)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=output_layer)

# %%
model.summary()
# %%
model2 = keras.models.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])
# %%
model2.summary()
# %%
w, b = model.weights
# %%
x = tf.linspace(-1.0, 1.0, 100)
x
# %% [markdown]
# #Build multi-layer model for regression

# %%
model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=(2,)),
    layers.Dense(units=3, activation='relu'),
    layers.Dense(units=1),
])
# %%
concrete = pd.read_csv("../data/concrete.csv")
concrete.head()

# %%
len(concrete.columns)
# %%
input_shape = (8,)
# %%
model = keras.models.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1, activation='linear'),
])

# %%
# we can also include activation as separate layers
model = keras.models.Sequential([
    layers.Dense(units=512, input_shape=input_shape),
    layers.Activation('relu'),
    layers.Dense(units=512),
    layers.Activation('relu'),
    layers.Dense(units=512),
    layers.Activation('relu'),
    layers.Dense(units=1),
])

# %%
# plot activationfunctions
activs = ['linear', 'relu', 'elu', 'selu']
activation_layers = [layers.Activation(a) for a in activs]

x = tf.linspace(-3.0, 3.0, 100)

fig, axs = plt.subplots(2, 2, dpi=300)
for i, j in itertools.product(range(2), range(2)):
    print(i, j)
    axs[i, j].plot(x, activation_layers[2*i+j](x))
    axs[i, j].set_xlabel('Input')
    axs[i, j].set_ylabel('Output')
fig.show()
# %% [markdown]
# learn with the cereals dataset


# %%
model.compile(
    optimizer="adam",
    loss="mae",
)
# %%
red_wine
# %%
# split into training and validation
seed = 42
df_train = red_wine.sample(frac=0.7, random_state=seed)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))  # type: ignore
# %%
# scale data between 0 and 1
max_ = df_train.max(axis=0)  # axis 0 means for every column we take the max
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


def denormalize(df):
    return (max_ - min_) * df + min


def denormalize_ans(y):
    return (max_.quality - min_.quality)*y + min_.quality


# %%
# split features and target
# axis 1  means for every row we drop 'quality'
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train.quality
y_valid = df_valid.quality
# %%
print(X_train.shape)
# %%
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=[11]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1)
])

# %%
model.compile(
    optimizer='adam',
    loss='mae',
)

# %%
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# %%
history_df = pd.DataFrame(history.history)
display(history_df)  # type: ignore
# %%
history_df.loss.plot()
# %%
fuel = pd.read_csv("../data/fuel.csv")

# %%
X = fuel.copy()
# remove target, fuel efficiency
y = X.pop('FE')
# %%
display(X)  # type: ignore
# %%
display(y)  # type: ignore
# %%
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

# %%
X = preprocessor.fit_transform(X)
y = np.log(y)
# %%
input_shape = (X.shape[1],)
print(f"Input shape: {input_shape}")
# %%
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])
# %%
model.compile(
    loss='mae',
    optimizer='adam',
)

# %%
history = model.fit(
    x=X,
    y=y,
    epochs=200,
    batch_size=128,
)

# %%
history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss']].plot()
# %%
learning_rate = 0.05
batch_size = 32
num_examples = 256
# %%
# models with different capacities
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

# %%
# early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)
# %%
seed = 42  # type: ignore
df_train = red_wine.sample(frac=0.7, random_state=seed)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))  # type: ignore
# %%
# scale data between 0 and 1
max_ = df_train.max(axis=0)  # axis 0 means for every column we take the max
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


def denormalize(df):
    return (max_ - min_) * df + min


def denormalize_ans(y):
    return (max_.quality - min_.quality)*y + min_.quality


# %%
# split features and target
# axis 1  means for every row we drop 'quality'
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train.quality
y_valid = df_valid.quality
# %%
print(X_train.shape)
# %%
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=[11]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1)
])

# %%
model.compile(
    optimizer='adam',
    loss='mae',
)

# %%
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping],
)
# %%
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
# %%
print(f"Minimum validation loss: {history_df['val_loss'].min()}")
# %%
spotify = pd.read_csv("../data/spotify.csv")
print(list(spotify.columns))
# %%
X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']
# %%
features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']
# %%
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)
# %%


def group_split(X, y, group, train_size=0.75):
    splitter = sklearn.model_selection.GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])


# %%
X_train, X_valid, y_train, y_valid = group_split(X, y, artists)
# %%
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.fit_transform(X_valid)
y_train = y_train / 100  # popularity is on a scale 0-100
y_valid = y_valid / 100
# %%
input_shape = [X_train.shape[1]]
print(f"Input shape: {input_shape}")
# %%
# First, start with a linear model
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape)
])

model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"Mimimum validation loss: {history_df.val_loss.min()}")
# %%
history_df.loc[10:, ['loss', 'val_loss']].plot()
print(f"Mimimum validation loss: {history_df.val_loss.min()}")

# %%
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"Mimimum validation loss: {history_df.val_loss.min()}")
# %% [markdown]
# Add early stoppage
# %% [markdown]
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)
early_stopping = keras.callbacks.EarlyStopping()
history = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"Mimimum validation loss: {history_df.val_loss.min()}")

# %%
