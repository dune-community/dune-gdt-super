from hapod.cellmodel.wrapper import CellModelSolver, DuneCellModel

if __name__ == "__main__":
    mu = {"Pa": 0.31622776601683794, "Be": 1, "Ca": 3.1622776601683795}
    solver = CellModelSolver("single_cell", 0.1, 0.001, 120, 120, 2, mu)
    m = DuneCellModel(solver)
    m.solve(mu=mu)
