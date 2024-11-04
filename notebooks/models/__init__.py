from models.lstm_model import LSTMModel
from models.multi_channel_lstm import MultiChannelLSTM
from models.cnn_lstm import CNN_LSTM
from models.attention_lstm import AttentionLSTM

def get_model_class(model_name):
    if model_name == 'lstm':
        return LSTMModel
    elif model_name == 'multi_channel_lstm':
         return MultiChannelLSTM
    elif model_name == 'cnn_lstm':
        return CNN_LSTM
    elif model_name == 'attention_lstm':
        return AttentionLSTM
    else:
        raise ValueError(f"Unknown model: {model_name}")
