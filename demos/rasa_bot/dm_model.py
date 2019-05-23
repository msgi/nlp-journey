import logging

from rasa_core.agent import Agent
from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy, FormPolicy
from rasa_core.policies.embedding_policy import EmbeddingPolicy

logger = logging.getLogger(__name__)


# dialogue manager model
def train_dialogue(domain_file='./domain/domain.yml',
                   model_path='./models/dialogue',
                   training_data_file='./stories/stories.md'):
    fallback = FallbackPolicy(fallback_action_name="utter_default",
                              core_threshold=0.2,
                              nlu_threshold=0.1)
    agent = Agent(domain_file,policies=[MemoizationPolicy(),
                                        KerasPolicy(),
                                        fallback,
                                        FormPolicy(),
                                        EmbeddingPolicy(epochs=100)])
    agent.visualize(training_data_file, output_file="graph.html", max_history=4)
    training_data = agent.load_data(training_data_file)  # augmentation_factor=0
    agent.train(training_data)
    agent.persist(model_path)
    return agent


if __name__ == '__main__':
    # train_dialogue()
    train_dialogue('./domain/domain_zh.yml', './models/dialogue_zh', './stories/stories_zh.md')
