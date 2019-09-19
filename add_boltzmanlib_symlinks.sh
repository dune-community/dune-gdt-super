if (($# > 0))
then
	build_type=$1
else
	build_type=gcc-release
fi
rm -f $(pwd)/boltzmann/libhapodgdt.so
rm -f $(pwd)/libhapodgdt.so
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/boltzmann/
