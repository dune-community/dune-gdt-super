#!/bin/bash
sed -i 's|#warning add list of neighbors ...|//#warning add list of neighbors ...|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|#warning only handle the newest intersections / merger info|//#warning only handle the newest intersections / merger info|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout << "GridGlue: Constructor succeeded!" << std::endl;|//std::cout << "GridGlue: Constructor succeeded!" << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout << ">>>> rank " << myrank << " coords: "|//std::cout << ">>>> rank " << myrank << " coords: "|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< patch0coords.size() << " and " << patch1coords.size() << std::endl;|//<< patch0coords.size() << " and " << patch1coords.size() << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout << ">>>> rank " << myrank << " entities: "|//std::cout << ">>>> rank " << myrank << " entities: "|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< patch0entities.size() << " and " << patch1entities.size() << std::endl;|//<< patch0entities.size() << " and " << patch1entities.size() << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout << ">>>> rank " << myrank << " types: "|//std::cout << ">>>> rank " << myrank << " types: "|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< patch0types.size() << " and " << patch1types.size() << std::endl;|//<< patch0types.size() << " and " << patch1types.size() << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout << myrank|//std::cout << myrank|g' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< " GridGlue::mergePatches : rank " << patch0rank << " / " << patch1rank << std::endl;|//<< " GridGlue::mergePatches : rank " << patch0rank << " / " << patch1rank << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< " GridGlue::mergePatches : "|//<< " GridGlue::mergePatches : "|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|<< "The number of remote intersections is " << intersections_.size()-1 << std::endl;|//<< "The number of remote intersections is " << intersections_.size()-1 << std::endl;|' dune-grid-glue/dune/grid-glue/adapter/gridglue.cc
sed -i 's|std::cout<<"ContactMerge building grid!\\n";|//std::cout<<"ContactMerge building grid!\\n";|' dune-grid-glue/dune/grid-glue/merging/contactmerge.hh
sed -i 's|std::cout << "StandardMerge building merged grid..." << std::endl;|//std::cout << "StandardMerge building merged grid..." << std::endl;|' dune-grid-glue/dune/grid-glue/merging/standardmerge.hh
sed -i 's|std::cout << "setup took " << watch.elapsed() << " seconds." << std::endl;|//std::cout << "setup took " << watch.elapsed() << " seconds." << std::endl;|' dune-grid-glue/dune/grid-glue/merging/standardmerge.hh
sed -i 's|std::cout << "intersection construction took " << watch.elapsed() << " seconds." << std::endl;|//std::cout << "intersection construction took " << watch.elapsed() << " seconds." << std::endl;|' dune-grid-glue/dune/grid-glue/merging/standardmerge.hh
sed -i 's|std::cout << "This is Codim1Extractor on a <" << dim|//std::cout << "This is Codim1Extractor on a <" << dim|' dune-grid-glue/dune/grid-glue/extractors/codim1extractor.hh
sed -i 's|<< "," << dimworld << "> grid!"|//<< "," << dimworld << "> grid!"|' dune-grid-glue/dune/grid-glue/extractors/codim1extractor.hh
sed -i 's|<< std::endl;|//<< std::endl;|' dune-grid-glue/dune/grid-glue/extractors/codim1extractor.hh
sed -i 's|std::cout << "added " << simplex_index << " subfaces\\n"|//std::cout << "added " << simplex_index << " subfaces\\n"|' dune-grid-glue/dune/grid-glue/extractors/codim1extractor.hh
