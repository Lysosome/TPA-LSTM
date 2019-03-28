import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def test(para, sess, model, data_generator):
    sess.run(data_generator.iterator.initializer)

    test_rse = 0.0
    count = 0
    n_samples = 0
    all_outputs, all_labels = [], []

    tp, fp, tn, fn = 0, 0, 0, 0

    ALPHA_FNAME = "./alphas.txt"
    # Fetch Embedding Weights (from num_features to embed_size)
    embedding_var = [v for v in tf.global_variables() if v.name == "model/dense/kernel:0"][0]
    # embed_weights (after transpose): [EMBED_SIZE, NUM_FEATURES]
    # embed_weights = np.abs(np.transpose(sess.run(embedding_var)))
    embed_weights = np.transpose(sess.run(embedding_var))

    # alphas and var_weights for EVERY batch
    all_alphas = []
    all_var_weights = []

    while True:
        try:
            # alphaNames = [model.rnn_inputs, model.rnn_inputs_embed]
            alphaNames = [n.name+":0" for n in tf.get_default_graph().as_graph_def().node
                     if (n.name.find("temporal_pattern_attention_cell_wrapper/attention")!=-1 and
                         n.name.find("Sigmoid")!=-1)]
                     # if (n.name == "model/rnn/cond/rnn/multi_rnn_cell/cell_0/cell_0/temporal_pattern_attention_cell_wrapper/attention/Reshape_2")]
            fetchList = [model.all_rnn_outputs[:, 0], model.labels]

            fetchList2 = [model.all_rnn_outputs_pre_reg, model.reg_outputs]
            fetchList.extend(fetchList2)

            fetchList.extend(alphaNames)
            results = sess.run( fetches=fetchList )

            outputs = results[0]
            labels = results[1]

            # each alpha shape: [BATCH_SIZE, EMBED_SIZE]
            alphas = results[2:]
            var_weights = []

            pre_reg = results[2]
            reg = results[3]
            # print("pre_reg (attention) outputs:", pre_reg[0][0] * data_generator.scale[0])
            # print("reg outputs:", reg[0][0] * data_generator.scale[0])
            # print("labels:", labels[0] * data_generator.scale[0])

            # var_weight shape: [BATCH_SIZE, NUM_FEATURES]
            # for alpha in alphas:
            #     var_weight = np.matmul(alpha, embed_weights)
            #     var_weights.append(var_weight)
            # all_alphas.append(alphas)
            # all_var_weights.append(var_weights)

            if para.mts:
                test_rse += np.sum(
                    ((outputs - labels) * data_generator.scale[0]) ** 2
                )
                all_outputs.append(outputs)
                all_labels.append(labels)
            elif para.data_set == 'muse' or para.data_set == 'lpd5':
                for b in range(para.batch_size):
                    for p in range(128):
                        if outputs[b][p] >= 0.5 and labels[b][p] >= 0.5:
                            tp += 1
                        elif outputs[b][p] >= 0.5 and labels[b][p] < 0.5:
                            fp += 1
                        elif outputs[b][p] < 0.5 and labels[b][p] < 0.5:
                            tn += 1
                        elif outputs[b][p] < 0.5 and labels[b][p] >= 0.5:
                            fn += 1
            count += 1
            n_samples += np.prod(outputs.shape)
        except Exception as e:
            # print("Exception:", e)
            break
    if para.mts:
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        sigma_outputs = all_outputs.std(axis=0)
        sigma_labels = all_labels.std(axis=0)
        mean_outputs = all_outputs.mean(axis=0)
        mean_labels = all_labels.mean(axis=0)
        idx = sigma_labels != 0
        test_corr = (
            (all_outputs - mean_outputs) * (all_labels - mean_labels)
        ).mean(axis=0) / (sigma_outputs * sigma_labels)
        test_corr = test_corr[idx].mean()
        test_rse = (
            np.sqrt(test_rse / n_samples) # / data_generator.rse
        )
        logging.info("test rse: %.5f, test corr: %.5f" % (test_rse, test_corr))

        # write alphas to file
        # print("Writing extracted var_weights to "+ALPHA_FNAME)
        # with open(ALPHA_FNAME, "w") as f:
        #     for i in range(len(all_var_weights)):
        #         f.write("BATCH "+(str)(i)+" ----------------------------\n\n")
        #         for j in range(len(alphas)):
        #             f.write(alphaNames[j]+"\n")
        #             f.write(np.array2string(all_var_weights[i][j], precision=4, threshold=np.nan))
        #             f.write("\n\n")

    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall >= 1e-6:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0.0
        logging.info('# of testing data: %d' % count * para.batch_size)
        logging.info('precision: %.5f' % precision)
        logging.info('recall: %.5f' % recall)
        logging.info('F1 score: %.5f' % F1)
