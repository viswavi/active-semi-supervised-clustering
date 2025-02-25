from tqdm import tqdm

from few_shot_clustering.active_semi_supervised_clustering.active_semi_clustering.exceptions import InconsistentConstraintsException

# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    print("Preprocessing constraints")
    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in tqdm(range(n)):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2 and (x1 not in cl_graph[x2] and x2 not in cl_graph[x1]):
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in tqdm(cl):
        for x in ml_graph[i]:
            if x not in ml_graph[j] and j not in ml_graph[x]:
                add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            if y not in ml_graph[i] and i not in ml_graph[y]:
                add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                if x not in ml_graph[y] and y not in ml_graph[x]:
                    add_both(cl_graph, x, y)

    _ = """
    for (i, j) in tqdm(cl[:150]):
        timer1 = time.perf_counter()
        for x in ml_graph[i]:
            timer1_2 = time.perf_counter()
            if x not in ml_graph[j] and j not in ml_graph[x]:
                timer1_3 = time.perf_counter()
                timer_dict["check_ml"] += timer1_3 - timer1_2
                add_both(cl_graph, x, j)
                timer_dict["add_both_cl"] += time.perf_counter() - timer1_3
        timer2 = time.perf_counter()
        for y in ml_graph[j]:
            timer2_2 = time.perf_counter()
            if y not in ml_graph[i] and i not in ml_graph[y]:
                timer2_3 = time.perf_counter()
                timer_dict["check_ml"] += timer2_3 - timer2_2
                add_both(cl_graph, i, y)
                timer_dict["add_both_cl"] += time.perf_counter() - timer2_3
        timer3 = time.perf_counter()
        for x in ml_graph[i]:
            for y in ml_graph[j]:
                timer3_2 = time.perf_counter()
                if x not in ml_graph[y] and y not in ml_graph[x]:
                    timer3_3 = time.perf_counter()
                    timer_dict["check_ml"] += timer3_3 - timer3_2
                    add_both(cl_graph, x, y)
                    timer_dict["add_both_cl"] += time.perf_counter() - timer3_3
        timer4 = time.perf_counter()
        timer_dict["head_ml_expansion"] += timer2 - timer1
        timer_dict["tail_ml_expansion"] += timer3 - timer2
        timer_dict["both_sides_expansion"] += timer4 - timer3

    timer_dict = {"add_both_cl": 0.0,
                    "check_ml": 0.0,
                    "head_ml_expansion": 0.0,
                    "tail_ml_expansion": 0.0,
                    "both_sides_expansion": 0.0}    

    """

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise InconsistentConstraintsException('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


def preprocess_constraints_no_transitive_closure(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    return ml_graph, cl_graph