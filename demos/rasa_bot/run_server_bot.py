from gevent.pywsgi import WSGIServer
from rasa_core import server
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.utils import EndpointConfig


def run_server_bot(serve_forever=True,
                   model_path='./models/dialogue',
                   nlu_model_path='./models/nlu/default/current'):
    nlu_interpreter = RasaNLUInterpreter(nlu_model_path)
    action_endpoint = EndpointConfig(url="http://localhost:5055/webhook", serve_forever=serve_forever)
    agent = Agent.load(model_path, interpreter=nlu_interpreter, action_endpoint=action_endpoint)

    bot_app = server.create_app(agent)
    http_server = WSGIServer(('0.0.0.0', 5005), bot_app)
    http_server.start()
    http_server.serve_forever()
    return http_server


if __name__ == '__main__':
    run_server_bot(model_path='./models/dialogue_zh',
                   nlu_model_path='./models/nlu_zh/default/current')
