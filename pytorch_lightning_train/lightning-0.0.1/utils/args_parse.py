def add_default_args(parser, root_dir, possible_model_names, rand_seed):

    # tng, test, val check intervals
    parser.add_argument('--eval_test_set', dest='eval_test_set', action='store_true', help='true = run test set also')
    parser.add_argument('--max_nb_epochs', default=200, type=int, help='cap epochs')
    parser.add_argument('--min_nb_epochs', default=2, type=int, help='min epochs')

    # early stopping
    parser.add_argument('--disable_early_stop', dest='enable_early_stop', action='store_false')
    parser.add_argument('--early_stop_metric', default='val_acc', type=str)
    parser.add_argument('--early_stop_mode', default='min', type=str)
    parser.add_argument('--early_stop_patience', default=3, type=int, help='number of epochs until stop')

    # model saving
    parser.add_argument('--model_save_path', default=root_dir + '/model_weights')
    parser.add_argument('--model_save_monitor_value', default='val_acc')
    parser.add_argument('--model_save_monitor_mode', default='max')

    # model paths
    parser.add_argument('--model_load_weights_path', default=None, type=str)
    parser.add_argument('--model_name', default='', help=','.join(possible_model_names))

    # GPU
    parser.add_argument('--per_experiment_nb_gpus', default=1, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--single_run_gpu', dest='single_run_gpu', action='store_true')
    parser.add_argument('--disable_cuda', dest='disable_cuda', action='store_true')
