def evaluate_model(model, val_generator):
    loss, accuracy = model.evaluate(val_generator)
    print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')
