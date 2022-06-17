from parlai.core.torch_agent import TorchAgent, Output

class ParrotAgent(TorchAgent):
    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        # for each row in batch, convert tensor to back to text strings
        return Output([self.dict.vec2txt(row) for row in batch.text_vec])

    def build_model(self, batch):
        # Our agent doesn't have a real model, so we will return a placeholder
        # here.
        return None