import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("-z", '--zeros', 
                    required=True,
                    help="CSV file for zeros")

parser.add_argument("-p", '--poles', 
                    required=True,
                    help="CSV file for poles")

parser.add_argument("-k", '--gain', 
                    required=True,
                    help="CSV file for gain of filters")

parser.add_argument('--parent_dir', 
                    help="Parent directory where your CSV files are stored. Use if different from this script's directory.")


def convert_csv(args, dir):
    arrays = {}
    for name, path in vars(args).items():
        if name == 'parent_dir':
            continue
        dtype_ = float if name == 'gain' else complex
        data = np.genfromtxt(path, delimiter=',', dtype=dtype_)

        num_channels = len(data) if name == 'gain' else data.shape[1]
        arrays['len'] = num_channels # TODO this overwrites things a lot, change it / find smth better
        for ch in range(num_channels):
            # store without NaN values
            if name == 'gain':
                arrays[f'{name}_{ch}'] = data[ch][~np.isnan(data[ch])]
            else:
                arrays[f'{name}_{ch}'] = data[:, ch][~np.isnan(data[:, ch])]

                
    save_dir = os.path.join(dir, 'cochba_filters.npz')
    np.savez(save_dir, **arrays)

    print(f"Content of CSV files saved in {str(save_dir)}")
    return save_dir


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.parent_dir:
        # default working directory to script's directory
        file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(file_dir)
    else:
        os.chdir(args.parent_dir)


    result_path = convert_csv(args, os.getcwd())
    filters = np.load(result_path, allow_pickle=True)
    for item in filters:
        print(item)

