from train_model import train_model
from evaluate_model import evaluate_model
from data_preprocessing import get_data_generators
from config import TRAIN_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE

# Train the model
model, history = train_model(TRAIN_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)

# Save the trained model
model.save("trained_model.h5")  # Save in HDF5 format

# Evaluate the model
_, val_gen = get_data_generators(TRAIN_DIR, batch_size=BATCH_SIZE)
evaluate_model(model, val_gen)
