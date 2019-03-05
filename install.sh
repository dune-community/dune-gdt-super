git clone https://github.com/dune-community/dune-gdt-super.git hapod-dune-gdt-super
cd hapod-dune-gdt-super
git checkout hapod2
git submodule update --init --recursive
./setup_virtualenv.sh
CC=gcc ./bin/download_external_libraries.py
CC=gcc ./bin/build_external_libraries.py
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release.ninja all
./add_boltzmanlib_symlinks.sh gcc-release



