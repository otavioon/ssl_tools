import argparse

def ssl_pretext_parser():
    parser = argparse.ArgumentParser(description='SSL Pretext')
    parser.add_argument("--dataset", type=str, help="The path to the dataset")
    parser.add_argument("--output", type=str, help="The path to the output")
    