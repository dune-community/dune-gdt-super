if (($# > 0))
then
	build_type=$1
else
	build_type=gcc-release
fi
rm -f $(pwd)/hapod/boltzmann/libhapodgdt.so
rm -f $(pwd)/hapod/cellmodel/libhapodgdt.so
rm -f $(pwd)/hapod/libhapodgdt.so
rm -f $(pwd)/libhapodgdt.so
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/hapod/boltzmann/
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/hapod/cellmodel/
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libhapodgdt.so $(pwd)/hapod/
