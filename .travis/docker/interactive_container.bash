DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"

docker run -it -v ${DIR}/dune-gdt:/root/src/dune-gdt dunecommunity/dune-gdt-testing:gcc-5_master
