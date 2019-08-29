if (($# > 0))
then
	build_type=$1
else
	build_type=gcc-release
fi
rm -f $(pwd)/boltzmann/libboltzmann.so
rm -f $(pwd)/libboltzmann.so
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libboltzmann.so $(pwd)/
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libboltzmann.so $(pwd)/boltzmann/
