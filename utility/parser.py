import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HFCL.")

    # ******************************   Optimizer Arguments      ***************************** #
    parser.add_argument('--lr', type=float, default=0.001,  # Common Parameters
                        help='Learning rate.')

    parser.add_argument('--lr_decay', action="store_true")
    parser.add_argument('--lr_decay_step', type=int, default=20)
    parser.add_argument('--lr_gamma', type=float, default=0.8)

    parser.add_argument('--test_epoch', type=int, default=5,
                        help='Number of epochs for testing.')
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Path to save models.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose dataset {Beibei, Taobao}.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval for evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                        help='Whether to normalize.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of iterations.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding dimension.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64,64]',
                        help='Output size of each layer.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization terms.')

    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify type of adjacency matrix (Laplacian matrix) {plain, norm, mean}.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')

    parser.add_argument('--Ks', nargs='?', default='[10, 50]',
                        help='Values of K in the Top-K list')


    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify test type {part, full}, whether to conduct citation on a small batch.')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable sparse level performance report, 1: Show sparse level performance report')

    parser.add_argument('--aux_beh_idx', nargs='?', default='[0,1]')

    # ******************************   SSL(CL) Loss Arguments      ***************************** #
    parser.add_argument('--aug_type', type=int, default=0)
    parser.add_argument('--ssl_ratio', type=float, default=0.5)
    parser.add_argument('--ssl_temp', type=float, default=0.2)  # Behavior-aware


    parser.add_argument('--ssl_reg', type=float, default=1)
    parser.add_argument('--ssl_mode', type=str, default='both_side')

    parser.add_argument('--ssl_reg_inter', nargs='?', default='[1,1,1]')
    parser.add_argument('--ssl_inter_mode', type=str, default='both_side')

    # ******************************   Model Hyperparameters      ***************************** #
    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='Negative weights, Beibei uses [0.1,0.1,0.1], Taobao uses [0.01,0.01,0.01]')

    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization term, Beibei uses 10, Taobao uses 0.01')  # Regularization term decay


    parser.add_argument('--coefficient', nargs='?', default='[0.0/6, 5.0/6, 1.0/6]',
                        help='Regularization term, Beibei uses [0.0/6, 5.0/6, 1.0/6], Taobao uses [1.0/6, 4.0/6, 1.0/6]')

    parser.add_argument('--cl_coefficient', type=int, default=1,
                        help='Contrastive learning loss coefficient')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.2]',
                        help='Message random dropout retention probability')

    parser.add_argument('--num_intents', type=int, default=16,
                        help='The number of intents, beibei:16  Taobao:12')

    return parser.parse_args()
