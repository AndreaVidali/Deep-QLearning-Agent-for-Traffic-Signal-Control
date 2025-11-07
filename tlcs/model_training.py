import numpy as np


def replay(model, memory, gamma, num_states, num_actions):
    """
    Retrieve a group of samples from the memory and for each of them update the learning equation, then train
    """
    batch = memory.get_samples(model.batch_size)

    if len(batch) > 0:  # if the memory is full enough
        # extract states from the batch
        states = np.array([val[0] for val in batch])

        # extract next states from the batch
        next_states = np.array([val[3] for val in batch])

        # predict Q(state), for every sample
        q_s_a = model.predict_batch(states)

        # predict Q(next_state), for every sample
        q_s_a_d = model.predict_batch(next_states)

        # setup training arrays
        x = np.zeros((len(batch), num_states))
        y = np.zeros((len(batch), num_actions))

        for i, b in enumerate(batch):
            # extract data from one sample
            state, action, reward, _ = (b[0], b[1], b[2], b[3])
            # get the Q(state) predicted before
            current_q = q_s_a[i]
            # update Q(state, action)
            current_q[action] = reward + gamma * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q  # Q(state) that includes the updated action value

        model.train_batch(x, y)  # train the NN
