if (($# > 0))
then
	build_type=$1
else
	build_type=gcc-release
fi

rm -f $(pwd)/gdt/cellmodel.so
ln -s $(pwd)/build/${build_type}/dune-gdt/lib/libcellmodel.so $(pwd)/gdt/cellmodel.so
