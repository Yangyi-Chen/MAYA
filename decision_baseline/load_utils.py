import torch

def load_victim_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    return model


def load_dataset(data_path):
    pass


def load_pos_tags(tag_path):
    pass


def load_word_candidates(candidates_path):
    pass




