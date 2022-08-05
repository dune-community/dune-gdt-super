from dune.gdt import visualize_discrete_functions_on_dd_grid, DiscreteFunction

def visualize_dd_functions(dd_grid, local_spaces, u, subdomains=None, name='u_ipdg'):
    discrete_functions = []
    subdomains = subdomains or list(range(dd_grid.num_subdomains))
    for I in subdomains:
        u_list_vector_array = u.block(I)
        u_I_istl = u_list_vector_array._list[0].real_part.impl
        u_I = DiscreteFunction(local_spaces[I], u_I_istl, name=name)
        discrete_functions.append(u_I)

    _ = visualize_discrete_functions_on_dd_grid(discrete_functions, dd_grid, subdomains)
