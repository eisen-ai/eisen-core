class GetShape:
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            print('Data stored in data dictionary at field {} has shape {}'.format(
                field,
                data[field].shape
            ))

        return data
