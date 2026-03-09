def encode_features(cur_phase, queues, queue_clip=20):
    phase_onehot = [0.0, 0.0, 0.0, 0.0]
    phase_onehot[cur_phase] = 1.0
    q_feats = [min(float(q), queue_clip) / float(queue_clip) for q in queues]
    return phase_onehot + q_feats
