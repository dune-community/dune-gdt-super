#!/bin/bash
sed -i 's;std::cout <<;// std::cout <<;g' dune-alugrid/dune/alugrid/common/defaultindexsets.hh
sed -i 's/const bool verbose = verb && this->comm().rank() == 0;/const bool verbose = false;\/\/verb && this->comm().rank() == 0;/g' dune-alugrid/dune/alugrid/3d/alugrid.hh
sed -i 's;msg {{.*}};msg;g' dune-alugrid/dune/alugrid/impl/serial/walk.h
sed -i 's|std::cerr << "WARNING (ignored): Could not open file|// std::cerr << "WARNING (ignored): Could not open file|g' dune-alugrid/dune/alugrid/impl/parallel/gitter_pll_sti.cc
sed -i 's|std::cerr << _ldbUnder|// std::cerr << _ldbUnder|g' dune-alugrid/dune/alugrid/impl/parallel/gitter_pll_sti.cc
sed -i 's|std::cout << dgfstream.str() << std::endl;|//std::cout << dgfstream.str() << std::endl;|' dune-alugrid/dune/alugrid/common/structuredgridfactory.hh
