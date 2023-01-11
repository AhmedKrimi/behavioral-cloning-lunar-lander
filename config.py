
class Config:
    # Agent type
    agent_type = "FCN"  # FCN or CNN
    hidden_units = 512
    # Frames
    skip_frames = 0
    history_length = 5
    # Optimzation
    lr = 0.0001
    batch_size = 512
    n_minibatches = 140000
    # testing
    n_test_episodes = 15
    rendering = True
