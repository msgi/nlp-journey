from nlp.chatbot.bot.chatbot_admin import ChatBot

if __name__ == '__main__':
    train_file = 'data/corpus/train.txt'
    model_path = 'model/bot/model_weights.h5'
    decoder_vector_path = 'model/bot/decoder_vector.m'
    encoder_vector_path = 'model/bot/encoder_vector.m'
    chatbot = ChatBot(train_file, model_path, decoder_vector_path,encoder_vector_path)
    print(chatbot.predict(['你好']))
