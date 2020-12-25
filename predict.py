import argparse
import pickle
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default='model.pkl')
    parser.add_argument('--json_path', default='example.json')
    args = parser.parse_args()

    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    record = json.loads(open(args.json_path, 'r').read())
    print(model.predict(np.array(list(record.values())).reshape(1, -1))[0])

if __name__ == "__main__":
    main()