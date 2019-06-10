import os
from bert_base.train.train_helper import get_args_parser
from bert_base.train.bert_lstm_ner import train

args = get_args_parser()
if True:
    import sys
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

# for i in dir(args):
#     print(i)
# print(type(dir(args)))
# print(type(vars(args)))
# for key, value in vars(args).items():
#     print(key, value)

    
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
# train(args=args)