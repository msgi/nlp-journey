from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.utils import EndpointConfig


def run_bot(serve_forever=True,
            model_path='./models/dialogue',
            nlu_model_path='./models/nlu/default/current'):
    nlu_interpreter = RasaNLUInterpreter(nlu_model_path)
    action_endpoint = EndpointConfig(url="http://localhost:5055/webhook", serve_forever=serve_forever)
    agent = Agent.load(model_path, interpreter=nlu_interpreter, action_endpoint=action_endpoint)
    return agent


if __name__ == '__main__':
    # agent = run_bot()
    agent = run_bot(model_path='./models/dialogue_zh',
                    nlu_model_path='./models/nlu_zh/default/current')
    print("Your bot is ready to talk! Type your messages here or send 'stop'")
    while True:
        a = input()
        if a == 'stop':
            break
        responses = agent.handle_text(a)
        for response in responses:
            print(response["text"])
