def encode_features(
    cur_phase,
    queues,
    avg_waits,
    max_waits,
    downstream_totals,
    queue_clip=20,
    wait_clip=60,
    downstream_clip=30,
):
    phase_onehot = [0.0, 0.0, 0.0, 0.0]
    phase_onehot[cur_phase] = 1.0

    q_feats = [min(float(q), queue_clip) / float(queue_clip) for q in queues]
    avg_feats = [min(float(w), wait_clip) / float(wait_clip) for w in avg_waits]
    max_feats = [min(float(w), wait_clip) / float(wait_clip) for w in max_waits]
    downstream_feats = [
        min(float(q), downstream_clip) / float(downstream_clip)
        for q in downstream_totals
    ]

    return phase_onehot + q_feats + avg_feats + max_feats + downstream_feats
