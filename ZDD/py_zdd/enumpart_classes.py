class Node:
    def __init__(self, edge, cc=0, comp=set(), fps=set()):
        self.edge = edge
        self.cc = cc
        self.comp = comp
        self.fps = fps
        self.paths = 0

    def summary(self):
        print(f" -- Node {self.edge}")
        print(f" --   cc {self.cc}")
        print(f" -- comp {self.comp}")
        print(f" --  fps {self.fps}")
        return
