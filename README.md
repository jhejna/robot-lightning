# Robot Lightning

This is accompanying code for the (research-lightning)[https://github.com/jhejna/research-lightning] repository. It provides a number of different robot controllers which interface with the robot environments in research lightning.


## Desktop Setup
On the desktop computer, first create a conda environment with the desired dependencies for policy inference. Afterwards, we will setup the dependencies for robot-lightning.

First, install the library dependencies via pip:
```
pip install -r requirements.txt
```
Or if using conda, you may want to add these to your conda environment file.


### Oculus Reader
If you want to use VR, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader).
1. Install `git lfs` via pip instead of following the sudo setup instructions. This can be done with `pip install git-lfs`.
2. Install adb devices: `sudo apt install android-tools-adb`.
3. After doing this, you can clone the oculus reader repository with `git clone https://github.com/rail-berkeley/oculus_reader`.
4. Then, install the package from the oculus reader repo with `pip install -e .`


### Demonstration collection
To collect demos, run
```
python scripts/collect_demos.py --path path/to/save/ --config path/to/robot/config
```

If the oculus isn't working due to adb permissions, run the following:
```
sudo adb kill-server
sudo adb start-server
```

## NUC Setup

On the NUC, first make sure to install the intel realtime kernel. Then, follow the setup instructions for individual packages.

### Polymetis
We use the monometis repo found [here](https://github.com/hengyuan-hu/monometis/tree/main).

1. Create the monometis conda environment by running `conda env create -f environments/nuc_monometis_env.yaml`.
2. Clone the monometis repo: `git clone https://github.com/hengyuan-hu/monometis/tree/main`.
3. Compile the library, following these instructions:
```
./scripts/build_libfranka.sh
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON
make -j
cd ../..
```
4. Install polymetis: `pip install -e ./polymetis`

Then, you can start the polymetis server as you would normally.

If you want to use ZeroRPC to wrap polymetis, you need to additionally install the robots package, then run
```
python scripts/start_nuc_remote.py
```
from the the NUC.
