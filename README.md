```
# This file is part of the dune-gdt-super project:
#   https://github.com/dune-community/dune-gdt-super
# The copyright lies with the authors of this file (see below).
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```
Dieses Repository enthält den Code für die numerischen Experimente zum HAPOD2 Paper.

# Installieren und Ausführen auf normalem Rechner oder Jaina/Jacen/Syal/Myri etc:

## Installieren

```bash
git clone git@zivgitlab.uni-muenster.de:ag-ohlberger/dune-community/dune-gdt-super.git
cd dune-gdt-super
git checkout hapod2
git submodule update --init --recursive
CC=gcc ./bin/build_external_libraries.py
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release.ninja all
source build/gcc-release/dune-python-env/bin/activate
python3 -m pip install
python -m pip --no-cache-dir install numpy scipy cython mpi4py rich neovim
cd pymor
pip install -e .
cd ..
cd build/gcc-release/dune-xt
ninja && ninja bindings
cd ../dune-gdt
ninja && ninja bindings
./add_libhapodgdt_symlinks.sh
```

## Ausführen

Die HAPOD-DEIM kann ausgeführt werden mit
```bash
source build/gcc-release/dune-python-env/bin/activate
mpiexec -n 32 python3 cellmodel_hapod_deim.py testcase t_end dt grid_size_x grid_size_y pfield_tol ofield_tol stokes_tol pfield_deim_tol ofield_deim_tol stokes_deim_tol calculate_errors parameter_sampling_type pod_method visualize subsampling include_newton_stages
```
Die Ergebnisse finden sich dann im Ordner dune-gdt-super/logs als .txt Datei. Das Programm bricht am Ende immer mit einem MPI-Fehler ab, den habe ich auch nach langem Suchen einfach nicht beheben können bisher. Aber es scheint nur irgendein Problem beim Aufräumen zu sein, das Programm an sich läuft ohne Probleme.

Die Kommandozeilenargumente sind:
- testcase: Testcase, relevant sind nur die Optionen cell_isolation_experiment (das Hauptexperiment aus dem Paper) oder single_cell (einfache Runde Zelle, einmal im Paper benutzt zum Testen der Solver)
- t_end: Endzeitpunkt
- dt: Zeitschrittweite
- grid_size_x: Gittergröße in x-Richtung
- grid_siye_y: Gittergröße in y-Richtung
- pfield_tol: Prescribed mean l2 error für die Phasenfeld HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- ofield_tol: Prescribed mean l2 error für die Orientierungsfeld HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- stokes_tol: Prescribed mean l2 error für die Stokes HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- pfield_deim_tol: Prescribed mean l2 error für die Phasenfeld HAPOD für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- ofield_deim_tol: Prescribed mean l2 error für die Orientierungsfeld HAPOD  für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- stokes_deim_tol: Prescribed mean l2 error für die Stokes DEIM für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- calculate_errors: True oder False, wenn False, dann wird nur die HAPOD-DEIM gemacht, kein reduziertes Problem und damit auch keine Fehler berechnet
- parameter_sampling_type: Parameterverteilung (siehe unten), mögliche Werte: uniform, uniform_reciprocal, log, log_inverted, log_and_log_inverted
- pod_method: method_of_snapshots oder qr_svd (qr_svd ist viel langsamer, ich benutze immer method_of_snapshots)
- visualize: True oder False, ob das volle und das reduzierte Problem visualisiert werden sollen (für die Testparameter, als .vtu Dateien). Siehe visualize_step in der cellmodel_hapod_deim.py um festzulegen, dass jeder x-te Schritt visualisiert wird.
- subsampling: True oder False, ob subsampling beim Visualisieren verwendet werden soll
- include_newton_stages: True oder False, ob die Newton stages zur Basisgenerierung mit verwendet werden sollen (haben wir am Anfang getestet, war nicht notwendig, also jetzt immer False)

Zum Beispiel
```bash
mpiexec -n 32 python3 cellmodel_hapod_deim.py cell_isolation_experiment 1e-2 1e-3 40 40 1e-4 1e-4 1e-4 1e-11 1e-11 1e-11 True log_and_log_inverted method_of_snapshots False False False
```

Weitere Parameter im Quellcode (cellmodel_hapod_deim.py):
- excluded_params: Tuple mit Teilmenge von ("Be", "Ca", "Pa"), zum Beispiel ("Be",). Die enthaltenen Parameter werden nicht reduziert.
- rf: Faktor zwischen größtem und kleinstem Parameter, das heißt der Parameterbereich für alle Parameter ist [1/sqrt(rf), sqrt(rf)].
(Die Parameter sind normalisiert, deswegen ist 1 als Mittelwert ein sinnvoller Wert für alle Parameter)
Momentan ist rf by default auf 5.
- incremental_gramian: Ob die Gramian in der HAPOD inkrementell mit Daten aus dem letzten Schritt gebaut werden soll.
Hatten wir im HAPOD-Paper gemacht und gab da einen Speedup, aber hier hatte ich zwischendrin Probleme, weil das voraussetzt, dass die
Basis immer orthogonal (orthonormal?) ist, aber ich die Reorthonormalisierung in der HAPOD ausgeschaltet habe (weil das extrem lange dauert).
Momentan reorthogonalisieren wir nur einmal ganz am Ende die finale Basis. Darüber könnte man auch nochmal nachdenken, wie das am sinnvollsten ist.
Wenn man zwischendrin orthogonalisiert, hat man das Problem (außer der Performance), dass die Singulärwerte, die man in der HAPOD weiterverwendet,
nicht mehr zu den Vektoren passen (weil die Vektoren im Gram-Schmidt-Prozess eventuell verändert werden). Oder sehe ich das falsch?
Andererseits nehmen wir im HAPOD-Paper an, dass die Vektoren immer orthogonal sind, die aus den lokalen PODs kommen,
da wollte ich auch immer nochmal durchrechnen, ob das eigentlich notwendig ist (wenn ja, sollte man vielleicht doch
zwischendrin reorthogonalisieren). So wie es jetzt ist, klappt es ja aber momentan gut
(Keine inkrementelle Gramian, Reorthonormalisierung nur ganz am Ende).
pol_order: Finite-Elemente Polynomgrad (ich habe immer 2 genommen jetzt, kann einfach so bleiben)
chunk_size: Chunk size für die HAPOD (lasse ich immer auf 10, d.h. immer nach 10 Zeitschritten wird wieder ein lokale POD gemacht)
visualize_step: Schrittgröße für die Visualisierung, bei 1 wird jeder Zeitschritt visualisiert, bei 10 jeder 10..
least_squares_pfield, least_squares_ofield, least_squares_stokes: True oder False, ob Least squares oder
Galerkin-Projektion verwendet werden soll (eventuell funktioniert die Galerkin-Projektion aber gerade nicht einfach so,
wir benutzen ja eigentlich least squares.).
product_type: Inneres Produkt, sollte immer auf "l2" stehen (vielleicht geht auch None, aber vielleicht geht dann auch irgendwo der Visualisierungscode o.Ä. kaputt).
Entspricht aber der Übergabe von None an die POD etc.
Zwischendrin hatte ich ja L2 und H1 getestet, aber du hattest ja gesagt, dass das keinen Sinn macht (hat auch nichts verbessert).
Jetzt sollte immer das euklidische Produkt verwendet werden (auch wenn im Quellcode eventuell noch an einigen Stellen Sonderfälle für H1 und L2 sind).
- omega: Omega parameter der HAPOD (steht immer auf 0.95 zur Zeit)
- train_params_per_rank, test_params_per_rank: Wieviele Parameter pro MPI rank verwendet werden sollen.
Per default wird pro MPI rank ein Trainingsparameter und ein Testparameter verwendet.
Das würde ich auch so lassen, wenn es geht, zwischendrin habe ich mehrere Parameter pro Rank verwendet, ich habe aber das
Gefühl, dass da noch irgendwo ein Fehler drin ist (Vermutlich im Updaten der Parameter für den C++ solver)






