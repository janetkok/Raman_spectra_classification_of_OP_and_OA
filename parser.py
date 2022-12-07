import argparse
import yaml 

def parse_args_train():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(
        description='OP and OA classification')

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4, metavar='T',
        help='Number of threads to be used (default: 4)')
    parser.add_argument('--method', default='svm', type=str,
                        help='svm/dt.' )
    # Dataset 
    parser.add_argument('--dataset', '-d', metavar='DDIR', default='./dataset',
                        help='dataset') 
    parser.add_argument('--neurons', type=int, default=22, help='number of neurons in ANN')
    parser.add_argument('--num-feat', type=int, default=1015, help='number of features used')
    parser.add_argument('--channel', nargs='+', default=[], metavar='STD', help='Override std deviation of of dataset')
    # Model parameters
    parser.add_argument('--train-batch', type=int, default=6, metavar='TB',
        help='Training batch size (default: 6)')
    parser.add_argument('--valid-batch', type=int, default=2, metavar='VB',
        help='Validation batch size (default: 1)')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
        help='Learning rate (default: 0.001)')

    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='CH',
                        help='number of checkpoints to keep (default: 10)')


    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text