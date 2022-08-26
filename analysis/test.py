import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', nargs='+', type=str)
args = parser.parse_args()

print(args)

print(args.test)