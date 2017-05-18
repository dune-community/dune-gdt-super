DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"

docker run --rm -it -v ${DIR}/dune-gdt:/root/src/dune-gdt dunecommunity/dune-gdt-testing_gcc:master bash
