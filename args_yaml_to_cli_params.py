import yaml
import sys

def convert_to_cli_args(yaml_filepath, dataset):
    with open(yaml_filepath, 'r') as f:
        try:
            args_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    # Convert dictionary keys and values to CLI arguments
    cli_args = []
    for key, value in args_dict[dataset].items():
        cli_args.append('--' + key)
        cli_args.append(str(value))

    return cli_args

if __name__ == '__main__':
    yaml_filepath = sys.argv[1]
    dataset = sys.argv[2]
    cli_args = convert_to_cli_args(yaml_filepath, dataset)
    for arg in cli_args:
        print(arg, end=' ')
