import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', type=str, help='Path to load the config file', required=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    os.makedirs(name=args.output, exist_ok=True)

    if args.bigtiff:
        match_arrow_template_from_mapjson(args)
    else:
        match_arrow_template(args)