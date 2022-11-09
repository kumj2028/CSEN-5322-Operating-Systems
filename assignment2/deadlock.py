import sys
from collections import defaultdict, deque

class DeadLockDetected(Exception):
    pass

class RAG():
    def __init__(self, input, output):
        self.graph = defaultdict(list)
        self.resource_queues = defaultdict(deque)
        self.vertices = set()
        self.deadlock = False
        self.input = input
        self.output = output
    
    def needs(self, p, r):
        self.output.write(f"Process {p} needs resource {r} - ")
        if len(self.graph[r]) == 0:
            self.graph[r].append(p)
            self.output.write(f"Resource {r} is allocated to process {p}.\n")
        else:
            self.graph[p].append(r)
            self.resource_queues[r].append(p)
            self.output.write(f"Process {p} must wait.\n")
            cycle = self.find_cycle()
            if len(cycle[0]) > 0:
                self.output.write(f"DEADLOCK DETECTED: Processes {', '.join(cycle[0])} " +
                    f"and Resources {', '.join(cycle[1])} are found in a cycle.")
                self.deadlock = True
    
    def releases(self, p, r):
        self.output.write(f"Process {p} releases resource {r} - ")
        self.graph[r].remove(p)
        if len(self.resource_queues[r]) == 0:
            self.output.write(f"Resource {r} is now free.\n")
        else:
            oldest_p = self.resource_queues[r].popleft()
            self.graph[r].append(oldest_p)
            self.graph[oldest_p].remove(r)
            self.output.write(f"Resource {r} is allocated to process {oldest_p}.\n")

    def find_cycle(self):
        cycle = [[],[]]
        visited = {}
        finished = {}
        def dfs(v):
            if finished[v]:
                return
            if visited[v]:
                print(f"Cycle found: {cycle}")
                raise DeadLockDetected
            visited[v] = True
            if type(v) == int:
                cycle[0].append(str(v))
            else:
                cycle[1].append(v)
            for neighbor in self.graph[v]:
                dfs(neighbor)
            finished[v] = True
            if type(v) == int:
                cycle[0].remove(str(v))
            else:
                cycle[1].remove(v)

        for v in self.vertices:
            visited[v] = False
            finished[v] = False
        for v in self.vertices:
            try:
                dfs(v)
            except DeadLockDetected:
                return cycle      
        return cycle

    def process(self):
        for line in self.input:
            process_str, command, resource = line.split()
            process = int(process_str)
            self.vertices.add(process)
            self.vertices.add(resource)
            if command == 'N':
                self.needs(process, resource)
                if self.deadlock:
                    return
            if command == 'R':
                self.releases(process, resource)
        self.output.write("EXECUTION COMPLETED: No deadlock encountered.")

if __name__ == '__main__':
    input = open(sys.argv[1], 'r')
    output = open('output.txt', 'w')
    rag = RAG(input, output)
    rag.process()
    input.close()
    output.close()