import os
import argparse
import sys
from pprint import pprint


def clean_folder(argv):
    # -------------------- Parse Arguments -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--keep_last', type=bool, default=True)
    args = parser.parse_args(argv[1:])

    files_to_delete = []

    dirs = sorted([os.path.join(args.dir,d) for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir,d))])
    for d in dirs:
        files = sorted([(os.path.join(d,f), int(f.replace('.pkl', '').replace('itr_',''))) for f in os.listdir(d) if 'itr_' in f], key=lambda x: x[1])

        if args.keep_last:
            files_to_delete.extend([f for f, i in files[:-1]])
        else:
            files_to_delete.extend([f for f, i in files])

    pprint(files_to_delete)

    inp = input("Are you shure you want to delete the files? yes/no \n")
    delete_files = inp == 'yes' or inp == 'y'

    if delete_files:
        for f in files_to_delete:
            print('deleting', f)
            os.remove(f)

if __name__ == "__main__":
    clean_folder(sys.argv)