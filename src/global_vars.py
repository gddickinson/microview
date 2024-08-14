class GlobalVars:
    def __init__(self):
        self.m = None  # This will hold the MicroView instance

    def set_microview(self, microview):
        self.m = microview

g = GlobalVars()
