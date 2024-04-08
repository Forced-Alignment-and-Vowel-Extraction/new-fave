from collections import defaultdict

def blank():
    return  ""

class Speaker(defaultdict):
    def __init__(self, *args):
        super().__init__(blank, *args)