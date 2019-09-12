import argparse
import ConfigParser
import logging.config
from solver import *
from argparse import Namespace


class ExcuteExperiments:
    def __init__(self, s):
        self.solver = s

    def excute(self):
        result = self.solver.run()
        return result

def update_args(args, key, value):
    tmp_dict = dict()
    for k, v in vars(args).items():
        tmp_dict[k] = v
    tmp_dict[key] = value
    args_out = Namespace(**tmp_dict)
    return args_out

if __name__ == '__main__':
    cf = ConfigParser.ConfigParser()
    cf.read("./conf/default_setting.conf")
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str, default=cf.get("path", "root_path"), required=False, help='root_path')
    parser.add_argument('--input_data_type', type=str, default=cf.get("path", "input_data_type"), required=False, help='input_data_type')
    parser.add_argument('--output_path', type=str, default=cf.get("path", "output_path"), required=False, help='output_path')
    parser.add_argument('--log_conf_path', type=str, default=cf.get("path", "log_conf_path"), required=False, help='log_conf_path')

    parser.add_argument('--global_dimension', type=int, default=cf.getint("parameters", "global_dimension"), required=False, help='number of latent factors')
    parser.add_argument('--word_dimension', type=int, default=cf.getint("parameters", "word_dimension"), required=False, help='word_dimension')
    parser.add_argument('--batch_size', type=int, default=cf.getint("parameters", "batch_size"), required=False, help='batch_size')
    parser.add_argument('--K', type=int, default=cf.getint("parameters", "K"), required=False, help='K')
    parser.add_argument('--epoch', type=int, default=cf.getint("parameters", "epoch"), required=False, help='epoch')
    parser.add_argument('--learning_rate', type=float, default=float(cf.get("parameters", "learning_rate")), required=False, help='learning_rate')
    parser.add_argument('--reg', type=float, default=float(cf.get("parameters", "reg")), required=False, help='reg')
    parser.add_argument('--mode', type=str, default=cf.get("parameters", "mode"), required=False, help='reg')
    parser.add_argument('--merge', type=str, default=cf.get("parameters", "merge"), required=False, help='merge')
    parser.add_argument('--concat', type=int, default=int(cf.get("parameters", "concat")), required=False, help='concat')
    parser.add_argument('--item_review_combine', type=str, default=cf.get("parameters", "item_review_combine"), required=False, help='item_review_combine')
    parser.add_argument('--item_review_combine_c', type=float, default=float(cf.get("parameters", "item_review_combine_c")),
                        required=False, help='item_review_combine_c')
    parser.add_argument('--lmd', type=float, default=float(cf.get("parameters", "lmd")), required=False, help='lmd')
    parser.add_argument('--drop_out_rate', type=float, default=float(cf.get("parameters", "drop_out_rate")), required=False, help='drop_out_rate')

    args = parser.parse_args()

    logging.config.fileConfig(os.path.join(args.root_path, args.log_conf_path))
    args.logger = logging.getLogger()
    

    result_file = './results/DER_result_' + args.mode + args.input_data_type.split('/')[-1]
    f = open(result_file, 'wb')
    f.writelines(args.mode + ' parameters:')
    f.writelines(str(args))
    f.writelines('\n')
    f.writelines(args.mode + ' result:')
    f.writelines('\n')
        

    with tf.variable_scope(args.mode):
        args.namespace = args.mode
        s = Solver(args, 0)
        exp = ExcuteExperiments(s)
        r = exp.excute()
        f.writelines(str(r))
        f.writelines('\n')
    f.close()


    






