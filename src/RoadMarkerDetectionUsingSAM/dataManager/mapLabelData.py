class MapLabelData:
    def __init__(self, name, component):
        self.name = name
        self.component = component

    def data_to_format(self):
        format = {self.name: {
                  'label': [[int(x), int(y)] for (x, y) in self.component]
                }}
        return format