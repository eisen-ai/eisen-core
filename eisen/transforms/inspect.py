class PrintFieldShape:
    def __init__(self, field):
        self.field = field

    def __call__(self, data):
        print('Data stored in data dictionary at field {} has shape {}'.format(
            self.field,
            data[self.field].shape
        ))

        return data
