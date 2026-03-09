def encode_features(cur_phase, queues, avg_waits, max_waits, queue_clip=20, wait_clip=60):

    phase_onehot = [0.0, 0.0, 0.0, 0.0]
    phase_onehot[cur_phase] = 1.0

    q_feats = [min(float(q), queue_clip) / float(queue_clip) for q in queues]
    avg_feats = [min(float(w), wait_clip) / float(wait_clip) for w in avg_waits]
    max_feats = [min(float(w), wait_clip) / float(wait_clip) for w in max_waits]

    return phase_onehot + q_feats + avg_feats + max_feats
