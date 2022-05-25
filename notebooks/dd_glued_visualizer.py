from dune.gdt import visualize_discrete_functions_on_dd_grid, DiscreteFunction

def visualize_dd_functions(dd_grid, local_spaces, u, subdomains=None, name='u_ipdg'):
    discrete_functions = []
    subdomains = subdomains or list(range(dd_grid.num_subdomains))
    for ss in subdomains:
        u_list_vector_array = u.block(ss)
        u_ss_istl = u_list_vector_array._list[0].real_part.impl
        u_ss = DiscreteFunction(local_spaces[ss], u_ss_istl, name=name)
        discrete_functions.append(u_ss)

    _ = visualize_discrete_functions_on_dd_grid(discrete_functions, dd_grid, subdomains)
