from models.lstm_model import LSTMModel

def get_model_class(model_name):
    if model_name == 'lstm':
        return LSTMModel
    # elif model_name == 'modelalternative':
    #     return ModelAlternative
    else:
        raise ValueError(f"Unknown model name: {model_name}")
