from nlp.chatbot.bot.chatbot import ChatBot

if __name__ == '__main__':
    train_file = 'data/corpus/train.txt'
    model_path = 'model/model_weights.h5'
    chatbot = ChatBot(train_file, model_path)
    print(chatbot.predict(['不约而同相聚在宏村。']))
