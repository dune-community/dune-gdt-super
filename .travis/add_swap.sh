#/bin/bash
set -e
SWAP=$HOME/swap.img
free
dd if=/dev/zero of=${SWAP} bs=1024k count=1000
sudo -E chown root:root ${SWAP}
sudo -E chmod 0600 ${SWAP}
sudo -E mkswap ${SWAP}
sudo -E swapon ${SWAP} 
free
