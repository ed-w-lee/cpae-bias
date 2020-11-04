import json


def build_graph_unweighted(definitions):
    '''
    Builds a graph where nodes are words and an edge exists if at least one
    word exists in the definition of another.
    '''
    graph = {}

    for word, definition in definitions.items():
        if word not in graph:
            graph[word] = set()

        for word2 in definition[0]:
            graph[word].add(word2)
            if word2 not in graph:
                graph[word2] = set()
            graph[word2].add(word)

    for i in graph:
        graph[i] = sorted(list(graph[i]))

    return graph


if __name__ == "__main__":
    with open('cpae/data/en_wn_full/all.json', 'r') as f:
        definitions = json.load(f)

    graph = build_graph_unweighted(definitions)

    with open('results/unweighted_graph.json', 'w+') as f:
        json.dump(graph, f)
