import os
import argparse

#TODO: try other normal estimation models (GeoWizard etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)

    args = parser.parse_args()
    
    cwd = os.getcwd()
    os.chdir('/home/yanyunzhi/code/DSINE')
    os.system(f'python test_waymo.py --datadir {args.datadir}')
    os.chdir(cwd)

        
