from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data_preprocessing import get_data_generators
from model_architecture import build_model


def train_model(train_dir, epochs=20, batch_size=32, lr=1e-4):
    train_gen, val_gen = get_data_generators(train_dir, batch_size=batch_size)
    model = build_model(num_classes=4)

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[lr_scheduler, early_stopping]
    )

    return model, history
