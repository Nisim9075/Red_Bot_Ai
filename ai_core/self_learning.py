from database.database import save_interaction
from models.machine_learning import ml_model

class SelfLearning:
    def __init__(self):
        self.new_data = []

    def learn_from_user(self, user_input, bot_response):
        self.new_data.append((user_input, bot_response))
        save_interaction(user_input, bot_response)

    def retrain_model(self):
        if self.new_data:
            data, labels = zip(*self.new_data)
            ml_model.train(list(data), list(labels))
            self.new_data = []

self_learning = SelfLearning()
