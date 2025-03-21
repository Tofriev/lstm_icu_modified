from models.lstm_model import LSTMModel
from models.multi_channel_lstm_static import MultiChannelLSTMStatic
from models.cnn_lstm import CNN_LSTM
from models.attention_lstm import AttentionLSTM
from models.lstm_model_static import LSTMModelStatic

def get_model_class(model_name):
    if model_name == 'lstm':
        return LSTMModel
    elif model_name == 'multi_channel_lstm_static':
         return MultiChannelLSTMStatic
    elif model_name == 'cnn_lstm':
        return CNN_LSTM
    elif model_name == 'attention_lstm':
        return AttentionLSTM
    elif model_name == 'lstm_static':
        return LSTMModelStatic
    else:
        raise ValueError(f"Unknown model: {model_name}")
