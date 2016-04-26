#/bin/bash
set -e
SWAP=$HOME/swap.img
MB_SIZE=${1:-4000}
free -h
sudo -E dd if=/dev/zero of=${SWAP} bs=1M count=${MB_SIZE}
sudo -E chown root:root ${SWAP}
sudo -E chmod 0600 ${SWAP}
sudo -E mkswap ${SWAP}
sudo -E swapon ${SWAP} 
free -h
